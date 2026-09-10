"""Builds a frequency dose-response canary dataset for CanaryBench memorization
auditing.

Reuses Group A (utility: AG News + MNLI) and Group B (PII detection:
AI4Privacy) EXACTLY as they exist in `dataset/processed/` -- only Group C
(the canary insertions) is rebuilt, with six frequency tiers instead of the
original three, so that per-canary training frequency can be studied as a
dose-response variable while holding total canary volume roughly constant.

See README.md / build_instruction_dataset.py for the original three-tier
construction this extends. Canary construction logic (masked-text tag
cleanup, PII span alignment, prompt formatting) is imported unchanged from
build_instruction_dataset.py so the new canaries are byte-for-byte the same
format as the original ones.

This is the FROZEN generator for the dose-response dataset: once a run's
validation checks (secret uniqueness, tier-balance, tier allocation) pass,
the output is not regenerated -- re-running after seeing downstream training
results would confound the frequency variable with a re-sampled canary set.
"""

import argparse
import json
import random
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_instruction_dataset import (  # noqa: E402
    clean_text,
    extract_pii_candidates,
    build_secret_recall_prompt,
    create_canary_record,
)

SEED = 42
AI4PRIVACY_DATASET = "ai4privacy/open-pii-masking-500k-ai4privacy"

# (tier_label, unique_count, exposure). unique_count * exposure = insertions.
DOSE_TIERS: List[Tuple[str, int, int]] = [
    ("1x", 200, 1),
    ("2x", 150, 2),
    ("5x", 150, 5),
    ("10x", 120, 10),
    ("20x", 100, 20),
    ("50x", 80, 50),
]
CANARY_REFERENCE_COUNT = 1_000
BASE_POOL_MARGIN = 800

EXPECTED_ORIGINAL_TRAIN_TOTAL = 83_000
# Max allowed spread (percentage points) of any field type's per-tier share
# from its overall share across all tiers+reference. Tight because tiers are
# now built by explicit stratified allocation, not by hoping shuffle order
# cooperates -- see allocate_counts()/build_canaries().
FIELD_BALANCE_TOLERANCE_PP = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a six-tier frequency dose-response canary dataset."
    )
    parser.add_argument(
        "--source-dir",
        default=str(SCRIPT_DIR / "dataset" / "processed"),
        help="Directory containing the original combined_*.jsonl / canary_registry.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "dataset" / "dose_response"),
        help="Directory where the new dose-response dataset will be written.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def normalize_secret(value: str, field_type: Optional[str] = None) -> str:
    """Canonicalizes a secret value so that two differently-formatted
    strings referring to the same underlying PII (e.g. "(555) 123-4567" vs
    "555-123-4567", or "Name@Site.com" vs "name@site.com") collide during
    uniqueness checks/collision-avoidance instead of silently passing as
    distinct. Field type is used when known (during our own construction);
    for reserved secrets pulled from Group B (no field type attached) it is
    inferred from the string's shape.
    """
    v = str(value).strip()
    ft = (field_type or "").upper()
    is_email = ft == "EMAIL" or "@" in v
    if is_email:
        return "email:" + v.lower()
    is_phone = ft == "PHONE" or bool(re.fullmatch(r"[\d\s()+\-.]+", v))
    if is_phone:
        digits = re.sub(r"\D", "", v)
        if digits:
            return "phone:" + digits
    return "other:" + v.lower()


def load_reserved_values(source_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Collects source texts and normalized secret values that must stay out
    of the new canary pool: every Group B pii_detection record's source
    text/targets (train+val+test, unchanged and reused as-is) plus every
    source text/secret already used by the ORIGINAL canary registry. The
    latter isn't strictly required by the task, but keeps the new
    dose-response canary set fully independent of the prior three-tier
    experiment rather than accidentally overlapping it.
    """
    reserved_sources: Set[str] = set()
    reserved_secrets: Set[str] = set()

    for name in ("combined_train.jsonl", "combined_val.jsonl", "combined_test.jsonl"):
        path = source_dir / name
        if not path.exists():
            continue
        for record in load_jsonl(path):
            if record.get("group") != "B":
                continue
            input_text = str(record.get("input", ""))
            if "Text: " in input_text:
                reserved_sources.add(input_text.split("Text: ", 1)[1])
            for value in str(record.get("target", "")).split(";"):
                value = value.strip()
                if value:
                    reserved_secrets.add(normalize_secret(value))

    registry_path = source_dir / "canary_registry.json"
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as handle:
            old_registry = json.load(handle)
        for item in old_registry:
            reserved_sources.add(str(item.get("source_text", "")))
            reserved_secrets.add(normalize_secret(item.get("secret", ""), item.get("field_type")))

    return reserved_sources, reserved_secrets


def build_base_pool(
    seed: int,
    required_unique: int,
    margin: int,
    reserved_sources: Set[str],
) -> List[dict]:
    """Mirrors build_instruction_dataset.build_canary_base_pool: streams real
    AI4Privacy English rows in a fixed shuffled order, cleans the [LABEL_n]
    mask tags to [LABEL], and extracts aligned PII candidates (field type,
    value, prefix-before-value) for each row that has at least one usable
    candidate. Rows whose source text is already reserved by Group B or the
    old canary registry are skipped outright.
    """
    pool_target = required_unique + margin
    dataset = load_dataset(AI4PRIVACY_DATASET)
    en_data = dataset["train"].filter(lambda row: row["language"] == "en").shuffle(seed=seed + 1)

    bases: List[dict] = []
    for idx, row in enumerate(en_data):
        source_text = clean_text(row.get("source_text") or row.get("unmasked_text") or "")
        masked_text = clean_text(
            re.sub(r"\[([A-Za-z0-9_-]+?)_\d+\]", r"[\1]", str(row.get("masked_text") or ""))
        )
        if not source_text or not masked_text or source_text in reserved_sources:
            continue

        pii_candidates = []
        seen_candidates = set()
        for candidate in extract_pii_candidates(source_text, masked_text):
            key = (candidate["label"], candidate["value"], candidate["prefix"])
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            pii_candidates.append(candidate)

        if not pii_candidates:
            continue

        bases.append(
            {
                "base_id": f"dose_canary_base_{idx}",
                "source_record_id": row.get("uid"),
                "source_text": source_text,
                "masked_text": masked_text,
                "pii_candidates": pii_candidates,
            }
        )
        if len(bases) >= pool_target:
            break

    if len(bases) < required_unique:
        raise ValueError(
            f"Not enough canary base rows: need at least {required_unique}, found {len(bases)}"
        )
    return bases


def resolve_pool_to_buckets(bases: Sequence[dict], rng: random.Random) -> Dict[str, List[dict]]:
    """Resolves each base row to exactly one (base, candidate) pair -- one
    canary opportunity per source row, as in the original design -- then
    buckets those opportunities by field type, preserving the pool's shuffle
    order within each bucket. This lets build_canaries() draw a controlled
    number of EMAIL/PHONE/etc. canaries per tier instead of relying on
    however the field types happened to fall in a single sequential draw.
    """
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for base in bases:
        candidate = rng.choice(base["pii_candidates"])
        buckets[candidate["label"]].append({"base": base, "candidate": candidate})
    return buckets


def allocate_counts(total: int, proportions: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder apportionment of `total` unique canary slots across
    field types according to `proportions` (which should sum to ~1.0), so
    each tier's field-type mix matches the overall pool's mix as closely as
    integer counts allow.
    """
    raw = {ft: total * p for ft, p in proportions.items()}
    alloc = {ft: int(v) for ft, v in raw.items()}
    remainder = total - sum(alloc.values())
    order = sorted(proportions.keys(), key=lambda ft: raw[ft] - alloc[ft], reverse=True)
    for ft in order[:remainder]:
        alloc[ft] += 1
    return alloc


def build_canaries(
    pool: Sequence[dict],
    reserved_secrets: Set[str],
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """Builds member (across DOSE_TIERS) and reference canaries by drawing
    from field-type buckets in proportion to the overall pool's field-type
    mix (stratified, not just randomly hoped-for), so every tier's
    EMAIL/PHONE/etc. composition -- and therefore its average target/prompt
    length -- tracks every other tier's as closely as the data allows.
    Secret-value collisions are checked on NORMALIZED values (see
    normalize_secret) so formatting differences can't hide a duplicate,
    and no base row is ever used twice (each bucket item is consumed at
    most once, across members AND references).
    """
    rng = random.Random(seed)
    buckets = resolve_pool_to_buckets(pool, rng)
    for items in buckets.values():
        rng.shuffle(items)
    cursors = {ft: 0 for ft in buckets}
    total_pool = sum(len(items) for items in buckets.values())
    proportions = {ft: len(items) / total_pool for ft, items in buckets.items()}

    used_secrets: Set[str] = set()
    fallback_used = 0

    def take(field_type: str) -> Optional[dict]:
        items = buckets.get(field_type, [])
        i = cursors.get(field_type, len(items))
        while i < len(items):
            item = items[i]
            i += 1
            norm = normalize_secret(item["candidate"]["value"], item["candidate"]["label"])
            if norm not in used_secrets and norm not in reserved_secrets:
                cursors[field_type] = i
                used_secrets.add(norm)
                return item
        cursors[field_type] = i
        return None

    def take_any(preferred: str) -> dict:
        nonlocal fallback_used
        order = [preferred] + [ft for ft in buckets if ft != preferred]
        for ft in order:
            item = take(ft)
            if item is not None:
                if ft != preferred:
                    fallback_used += 1
                return item
        raise ValueError(
            "Ran out of canary base rows across all field types while building "
            "dose-response canaries. Increase BASE_POOL_MARGIN."
        )

    def word_tokens(text: str) -> int:
        return len(str(text).split())

    member_train: List[dict] = []
    member_eval: List[dict] = []
    registry: List[dict] = []
    tier_stats: List[dict] = []
    unique_idx = 0

    def build_tier(tier_label: Optional[str], unique_count: int, exposure: int, is_member: bool) -> dict:
        nonlocal unique_idx
        alloc = allocate_counts(unique_count, proportions)
        field_counts: Counter = Counter()
        targets: List[str] = []
        prompts: List[str] = []
        for field_type, count in alloc.items():
            for _ in range(count):
                item = take_any(field_type)
                base = item["base"]
                candidate = item["candidate"]
                resolved_field_type = candidate["label"]
                secret = candidate["value"]
                prefix = candidate["prefix"]

                if is_member:
                    canary_id = f"member_canary_{unique_idx}"
                    unique_idx += 1
                    member_eval.append(
                        create_canary_record(
                            canary_id=canary_id,
                            base=base,
                            label=resolved_field_type,
                            secret=secret,
                            prefix=prefix,
                            exposure=exposure,
                            split="member_eval",
                        )
                    )
                    for rep in range(exposure):
                        member_train.append(
                            create_canary_record(
                                canary_id=f"{canary_id}_rep{rep}",
                                base=base,
                                label=resolved_field_type,
                                secret=secret,
                                prefix=prefix,
                                exposure=exposure,
                                split="train",
                            )
                        )
                    prompts.append(build_secret_recall_prompt(prefix))
                else:
                    canary_id = f"reference_canary_{_reference_offset[0]}"
                    _reference_offset[0] += 1
                    reference_records.append(
                        create_canary_record(
                            canary_id=canary_id,
                            base=base,
                            label=resolved_field_type,
                            secret=secret,
                            prefix=prefix,
                            exposure=0,
                            split="reference",
                        )
                    )
                    prompts.append(build_secret_recall_prompt(prefix))

                registry.append(
                    make_registry_entry(
                        canary_id=canary_id,
                        is_member=is_member,
                        tier=tier_label,
                        base=base,
                        field_type=resolved_field_type,
                        secret=secret,
                        prefix=prefix,
                        exposure=exposure,
                    )
                )
                field_counts[resolved_field_type] += 1
                targets.append(secret)

        return {
            "tier": tier_label if tier_label is not None else "reference",
            "unique_count": unique_count,
            "exposure": exposure,
            "insertions": unique_count * exposure,
            "field_type_counts": dict(field_counts),
            "field_type_pct": {
                ft: round(100 * n / unique_count, 1) for ft, n in field_counts.items()
            },
            "mean_target_length_chars": round(statistics.mean(len(t) for t in targets), 2),
            "mean_target_length_tokens": round(statistics.mean(word_tokens(t) for t in targets), 2),
            "mean_prompt_length_chars": round(statistics.mean(len(p) for p in prompts), 2),
            "mean_prompt_length_tokens": round(statistics.mean(word_tokens(p) for p in prompts), 2),
        }

    reference_records: List[dict] = []
    _reference_offset = [0]

    for tier_label, unique_count, exposure in DOSE_TIERS:
        tier_stats.append(build_tier(tier_label, unique_count, exposure, is_member=True))

    tier_stats.append(build_tier(None, CANARY_REFERENCE_COUNT, 0, is_member=False))

    if fallback_used:
        print(
            f"  note: {fallback_used} canary(ies) fell back to a non-preferred field type "
            "because their preferred bucket was exhausted"
        )

    return member_train, member_eval, reference_records, registry, tier_stats


def make_registry_entry(
    canary_id: str,
    is_member: bool,
    tier: Optional[str],
    base: dict,
    field_type: str,
    secret: str,
    prefix: str,
    exposure: int,
) -> dict:
    return {
        "id": canary_id,
        "is_member": is_member,
        "tier": tier,
        "frequency": exposure,
        "exposure": exposure,
        "prefix": build_secret_recall_prompt(prefix),
        "secret": secret,
        "field_type": field_type,
        "masked_text": base["masked_text"],
        "source_text": base["source_text"],
        "source_row_id": base["base_id"],
        "source_record_id": base["source_record_id"],
    }


def check_secret_uniqueness(registry: Sequence[dict]) -> Tuple[bool, List[str]]:
    """No normalized EMAIL/PHONE/etc. value may repeat anywhere in the
    registry (across member tiers AND references) -- this catches the case
    where two different source rows happen to carry the same underlying
    secret in different formatting, which plain string equality on raw
    values would miss.
    """
    normed = [normalize_secret(item["secret"], item.get("field_type")) for item in registry]
    counts = Counter(normed)
    dupes = [key for key, n in counts.items() if n > 1]
    return len(dupes) == 0, dupes


def run_validations(
    member_eval: Sequence[dict],
    reference_records: Sequence[dict],
    written_canary_train_records: Sequence[dict],
    registry: Sequence[dict],
    tier_stats: Sequence[dict],
    total_train_records: int,
) -> bool:
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            ok = False

    member_ids = {r["id"] for r in member_eval}
    reference_ids = {r["id"] for r in reference_records}
    check(
        "No overlap between member and reference canary IDs",
        member_ids.isdisjoint(reference_ids),
        f"{len(member_ids & reference_ids)} overlapping ids",
    )

    uniqueness_ok, dupes = check_secret_uniqueness(registry)
    check(
        "Secret uniqueness across everything (normalized EMAIL/PHONE, all tiers + reference)",
        uniqueness_ok,
        f"{len(dupes)} normalized secret(s) shared across canaries" if dupes else "",
    )

    train_secrets_by_id = defaultdict(int)
    for record in written_canary_train_records:
        base_id = record["id"].rsplit("_rep", 1)[0]
        train_secrets_by_id[base_id] += 1

    reference_secrets = {r["target"] for r in reference_records}
    train_secrets = {r["target"] for r in written_canary_train_records}
    check(
        "No reference canary secrets present in combined_train.jsonl insertions",
        reference_secrets.isdisjoint(train_secrets),
        f"{len(reference_secrets & train_secrets)} overlapping secrets",
    )

    expected_exposure = {r["id"]: r["exposure"] for r in member_eval}
    mismatches = []
    for canary_id, expected in expected_exposure.items():
        actual = train_secrets_by_id.get(canary_id, 0)
        if actual != expected:
            mismatches.append((canary_id, expected, actual))
    check(
        "All member canary repetitions present in combined_train.jsonl with correct counts",
        len(mismatches) == 0,
        f"{len(mismatches)} mismatches" if mismatches else "",
    )

    print("\nFrozen tier allocation:")
    print(f"{'frequency':<10} {'unique canaries':>16} {'inserted rows':>14}")
    total_unique = 0
    total_inserted = 0
    for stat in tier_stats:
        if stat["tier"] == "reference":
            continue
        print(f"{stat['tier']:<10} {stat['unique_count']:>16} {stat['insertions']:>14}")
        total_unique += stat["unique_count"]
        total_inserted += stat["insertions"]
    print(f"{'TOTAL':<10} {total_unique:>16} {total_inserted:>14}")

    print("\nDifficulty balance per tier:")
    header = (
        f"{'tier':>10} | {'unique':>6} | {'EMAIL%':>7} | {'PHONE%':>7} | "
        f"{'other%':>7} | {'target_tok':>10} | {'prompt_tok':>10}"
    )
    print(header)
    print("-" * len(header))

    overall_counts: Counter = Counter()
    for stat in tier_stats:
        overall_counts.update(stat["field_type_counts"])
    overall_total = sum(overall_counts.values()) or 1
    all_field_types = sorted(overall_counts.keys())

    balance_ok = True
    worst_spread = 0.0
    for stat in tier_stats:
        counts = stat["field_type_counts"]
        total = sum(counts.values()) or 1
        pct = {ft: 100 * counts.get(ft, 0) / total for ft in all_field_types}
        other_pct = sum(v for ft, v in pct.items() if ft not in ("EMAIL", "PHONE"))
        print(
            f"{stat['tier']:>10} | {stat['unique_count']:>6} | "
            f"{pct.get('EMAIL', 0):>6.1f}% | {pct.get('PHONE', 0):>6.1f}% | "
            f"{other_pct:>6.1f}% | {stat['mean_target_length_tokens']:>10.2f} | "
            f"{stat['mean_prompt_length_tokens']:>10.2f}"
        )
        for field_type in all_field_types:
            tier_share = counts.get(field_type, 0) / total
            overall_share = overall_counts[field_type] / overall_total
            spread_pp = abs(tier_share - overall_share) * 100
            worst_spread = max(worst_spread, spread_pp)
            if spread_pp > FIELD_BALANCE_TOLERANCE_PP:
                balance_ok = False
    check(
        f"Field type distribution balanced across tiers (within {FIELD_BALANCE_TOLERANCE_PP:.0f}pp of overall share)",
        balance_ok,
        f"worst observed spread: {worst_spread:.1f}pp",
    )

    check(
        "Total training records approx. 83,000",
        abs(total_train_records - EXPECTED_ORIGINAL_TRAIN_TOTAL) <= 2_000,
        f"got {total_train_records}",
    )

    return ok


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    seed = args.seed
    random.seed(seed)

    print("Loading reserved (excluded) source texts/secrets from Group B + old canary registry...")
    reserved_sources, reserved_secrets = load_reserved_values(source_dir)
    print(f"  reserved source texts: {len(reserved_sources)}, reserved secrets: {len(reserved_secrets)}")

    total_unique = sum(count for _, count, _ in DOSE_TIERS) + CANARY_REFERENCE_COUNT
    print(f"Building canary base pool (target unique={total_unique}, margin={BASE_POOL_MARGIN})...")
    pool = build_base_pool(seed, total_unique, BASE_POOL_MARGIN, reserved_sources)
    print(f"  pool size: {len(pool)}")

    print("Constructing member and reference canaries across 6 frequency tiers (stratified by field type)...")
    member_train, member_eval, reference_records, registry, tier_stats = build_canaries(
        pool, reserved_secrets, seed
    )

    print("Loading original combined_train.jsonl to keep Groups A + B unchanged...")
    original_train = load_jsonl(source_dir / "combined_train.jsonl")
    group_ab_records = [r for r in original_train if r.get("group") in ("A", "B")]
    group_a_count = sum(1 for r in group_ab_records if r["group"] == "A")
    group_b_count = sum(1 for r in group_ab_records if r["group"] == "B")

    new_train_records = list(group_ab_records) + list(member_train)
    rng = random.Random(seed)
    rng.shuffle(new_train_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(new_train_records, output_dir / "combined_train.jsonl")
    shutil.copyfile(source_dir / "combined_val.jsonl", output_dir / "combined_val.jsonl")
    shutil.copyfile(source_dir / "combined_test.jsonl", output_dir / "combined_test.jsonl")
    save_jsonl(member_eval, output_dir / "member_canaries.jsonl")
    save_jsonl(reference_records, output_dir / "reference_canaries.jsonl")
    with (output_dir / "canary_registry.json").open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)

    member_tier_stats = [s for s in tier_stats if s["tier"] != "reference"]
    reference_stat = next((s for s in tier_stats if s["tier"] == "reference"), {})
    total_insertions = sum(s["insertions"] for s in member_tier_stats)

    stats = {
        "seed": seed,
        "frozen": True,
        "tier_allocation": [
            {"frequency": s["tier"], "unique_canaries": s["unique_count"], "inserted_rows": s["insertions"]}
            for s in member_tier_stats
        ]
        + [
            {
                "frequency": "TOTAL",
                "unique_canaries": sum(s["unique_count"] for s in member_tier_stats),
                "inserted_rows": total_insertions,
            }
        ],
        "tiers": member_tier_stats,
        "reference": reference_stat,
        "totals": {
            "unique_member_canaries": sum(count for _, count, _ in DOSE_TIERS),
            "total_canary_insertions": total_insertions,
            "reference_count": CANARY_REFERENCE_COUNT,
            "group_a_count": group_a_count,
            "group_b_count": group_b_count,
            "total_train_records": len(new_train_records),
        },
    }
    with (output_dir / "dose_response_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"\nWrote outputs to {output_dir}")
    print(
        f"Group A: {group_a_count}, Group B: {group_b_count}, "
        f"new Group C insertions: {total_insertions}, total train: {len(new_train_records)}"
    )

    print("\nRunning validation checks (re-reading written combined_train.jsonl)...")
    written_train = load_jsonl(output_dir / "combined_train.jsonl")
    written_canary_records = [r for r in written_train if r.get("group") == "C"]
    written_group_b_secrets = {
        value.strip()
        for r in written_train
        if r.get("group") == "B"
        for value in str(r.get("target", "")).split(";")
        if value.strip()
    }
    ok = run_validations(
        member_eval=member_eval,
        reference_records=reference_records,
        written_canary_train_records=written_canary_records,
        registry=registry,
        tier_stats=tier_stats,
        total_train_records=len(written_train),
    )
    reference_ids_in_train = {r["id"] for r in reference_records} & {r["id"] for r in written_train}
    if reference_ids_in_train:
        print(f"[FAIL] Reference canary ids found in combined_train.jsonl: {len(reference_ids_in_train)}")
        ok = False
    reference_secrets_in_group_b = {r["target"] for r in reference_records} & written_group_b_secrets
    if reference_secrets_in_group_b:
        print(
            f"[FAIL] {len(reference_secrets_in_group_b)} reference canary secrets coincide with "
            "Group B target values in combined_train.jsonl"
        )
        ok = False
    if not ok:
        print("\nOne or more validation checks FAILED -- see above. Dataset NOT frozen; fix and rerun.")
        sys.exit(1)
    print("\nAll validation checks passed. Dataset is frozen -- do not regenerate.")


if __name__ == "__main__":
    main()

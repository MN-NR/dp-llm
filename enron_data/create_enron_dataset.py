import argparse
import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_VAL_SIZE = 1_000
DEFAULT_TEST_SIZE = 1_000
DEFAULT_CANARY_SOURCE_POOL_SIZE = 5_000
DEFAULT_CANARY_MEMBER_CONFIG = [(7000, 1), (2000, 10), (2500, 50)]
DEFAULT_CANARY_REFERENCE = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Enron LM dataset with canaries patched into real email prefixes."
    )
    parser.add_argument(
        "--input",
        help="Path to a directory of Enron emails or a JSONL/text file containing raw emails.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="corbt/enron-emails",
        help="Hugging Face dataset id to load when --input is not provided.",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split to load from the Hugging Face dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "processed"),
        help="Directory where the processed JSONL files will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--canary-source-pool-size",
        type=int,
        default=DEFAULT_CANARY_SOURCE_POOL_SIZE,
        help="Number of held-out emails reserved only for canary prefix extraction.",
    )
    parser.add_argument(
        "--canary-reference",
        type=int,
        default=DEFAULT_CANARY_REFERENCE,
    )
    parser.add_argument(
        "--member-config",
        default="7000x1,2000x10,2500x50",
        help="Comma-separated TOTALxEXPOSURE tiers, for example '7000x1,2000x10,2500x50'.",
    )
    parser.add_argument("--min-prefix", type=int, default=80)
    parser.add_argument("--max-prefix", type=int, default=300)
    parser.add_argument("--max-email-chars", type=int, default=1500)
    parser.add_argument(
        "--aux-train-path",
        help="Optional JSONL dataset to append to the Enron training split.",
    )
    parser.add_argument(
        "--aux-val-path",
        help="Optional JSONL dataset to append to the Enron validation split.",
    )
    parser.add_argument(
        "--aux-test-path",
        help="Optional JSONL dataset to append to the Enron test split.",
    )
    parser.add_argument(
        "--aux-train-limit",
        type=int,
        default=None,
        help="Optional max number of auxiliary training records to mix in.",
    )
    parser.add_argument(
        "--aux-val-limit",
        type=int,
        default=None,
        help="Optional max number of auxiliary validation records to mix in.",
    )
    parser.add_argument(
        "--aux-test-limit",
        type=int,
        default=None,
        help="Optional max number of auxiliary test records to mix in.",
    )
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("create_enron_dataset")


def save_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_jsonl_records(
    path: Path,
    split: str,
    logger: logging.Logger,
    limit: Optional[int] = None,
) -> List[dict]:
    records: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["split"] = split
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    logger.info("Loaded %s auxiliary %s records from %s", f"{len(records):,}", split, path)
    return records


def parse_member_config(config_text: str) -> List[Tuple[int, int]]:
    tiers: List[Tuple[int, int]] = []
    for chunk in config_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        total_text, exposure_text = chunk.split("x", 1)
        tiers.append((int(total_text), int(exposure_text)))
    if not tiers:
        raise ValueError("member-config must contain at least one TOTALxEXPOSURE tier")
    return tiers


def clean_email_text(text: str, max_len: int) -> str:
    text = (text or "").strip()
    text = re.sub(r"(?im)^message-id:.*$", " ", text)
    text = re.sub(r"(?im)^x-[^\\n]*$", " ", text)
    text = re.sub(r"(?im)^mime-version:.*$", " ", text)
    text = re.sub(r"(?im)^content-type:.*$", " ", text)
    text = re.sub(r"(?im)^content-transfer-encoding:.*$", " ", text)
    text = re.sub(r"(?im)^bcc:.*$", " ", text)
    text = re.sub(r"(?im)^cc:.*$", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:max_len].strip()


def iter_email_texts(input_path: Path) -> Iterable[str]:
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if path.is_file():
                try:
                    yield path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
        return

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield (
                    record.get("text")
                    or record.get("target")
                    or record.get("body")
                    or record.get("email")
                    or ""
                )
        return

    with input_path.open(encoding="utf-8", errors="ignore") as handle:
        yield handle.read()


def load_emails(input_path: Path, max_email_chars: int) -> List[str]:
    emails = []
    for raw_text in iter_email_texts(input_path):
        cleaned = clean_email_text(raw_text, max_len=max_email_chars)
        if cleaned:
            emails.append(cleaned)
    return emails


def build_email_from_hf_row(row: dict) -> str:
    parts = []
    sender = row.get("from") or ""
    subject = row.get("subject") or ""
    body = row.get("body") or ""

    recipients = row.get("to") or []
    if isinstance(recipients, str):
        recipients = [recipients]
    recipients = [item for item in recipients if item]

    if sender:
        parts.append(f"From: {sender}")
    if recipients:
        parts.append(f"To: {', '.join(recipients)}")
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append("")
        parts.append(body)
    return "\n".join(parts)


def load_emails_from_hf_dataset(
    dataset_name: str,
    split: str,
    max_email_chars: int,
    logger: logging.Logger,
) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading from Hugging Face requires the `datasets` package."
        ) from exc

    logger.info("Loading Hugging Face dataset %s [%s]", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    emails = []
    seen = set()
    for row in dataset:
        raw_text = build_email_from_hf_row(row)
        cleaned = clean_email_text(raw_text, max_len=max_email_chars)
        if not cleaned or len(cleaned) < 40:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        emails.append(cleaned)
    return emails


def load_raw_emails(
    input_path: Optional[Path],
    hf_dataset: str,
    hf_split: str,
    max_email_chars: int,
    logger: logging.Logger,
) -> List[str]:
    if input_path is not None:
        logger.info("Loading raw emails from %s", input_path)
        return load_emails(input_path, max_email_chars=max_email_chars)
    return load_emails_from_hf_dataset(
        dataset_name=hf_dataset,
        split=hf_split,
        max_email_chars=max_email_chars,
        logger=logger,
    )


def build_lm_record(record_id: str, text: str, split: str, source: str, **extra: object) -> dict:
    record = {
        "id": record_id,
        "group": "D",
        "task": "lm",
        "input": "",
        "target": text,
        "split": split,
        "source": source,
        "canary": False,
    }
    record.update(extra)
    return record


def generate_secret(length: int = 16) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def extract_prefix_from_email(email_text: str, min_prefix: int, max_prefix: int) -> str:
    if len(email_text) <= min_prefix:
        return email_text

    upper = min(max_prefix, len(email_text) - 1)
    if upper <= min_prefix:
        return email_text[:upper]

    cut = random.randint(min_prefix, upper)
    while cut < len(email_text) - 1 and email_text[cut].isalnum():
        cut += 1
    return email_text[:cut].strip()


def patch_secret_into_prefix(prefix: str, secret: str) -> str:
    suffix_templates = [
        "\n\nReference number: {secret}",
        "\n\nConfirmation code: {secret}",
        "\n\nTracking ID: {secret}",
        "\n\nAccount number: {secret}",
        "\n\nVerification code: {secret}",
        "\n\nPlease use access code {secret} for the portal.",
        "\n\nFor your records, the internal code is {secret}.",
        "\n\nWire details follow.\nReference: {secret}",
    ]
    suffix = random.choice(suffix_templates).format(secret=secret)
    return (prefix.rstrip() + suffix).strip()


def generate_canary_from_dataset_prefix(
    canary_id: str,
    source_text: str,
    secret_length: int,
    min_prefix: int,
    max_prefix: int,
) -> Tuple[dict, dict]:
    secret = generate_secret(secret_length)
    prefix = extract_prefix_from_email(
        source_text,
        min_prefix=min_prefix,
        max_prefix=max_prefix,
    )
    full_text = patch_secret_into_prefix(prefix, secret)

    record = {
        "id": canary_id,
        "group": "D",
        "task": "lm",
        "input": "",
        "target": full_text,
        "canary": True,
        "source": "canary_from_dataset",
        "prefix": prefix,
    }

    registry_entry = {
        "id": canary_id,
        "is_member": None,
        "secret": secret,
        "prefix": prefix,
        "full_text": full_text,
        "exposure": None,
    }
    return record, registry_entry


def build_canary_eval_record(
    record: dict,
    secret: str,
    split: str,
    exposure: int,
) -> dict:
    return {
        "id": record["id"],
        "group": record["group"],
        "task": "lm",
        "input": record["prefix"],
        "target": secret,
        "split": split,
        "canary": True,
        "source": record["source"],
        "prefix": record["prefix"],
        "full_text": record["target"],
        "exposure": exposure,
    }


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    random.seed(args.seed)

    input_path = Path(args.input) if args.input else None
    output_dir = Path(args.output_dir)
    member_config = parse_member_config(args.member_config)

    raw_emails = load_raw_emails(
        input_path=input_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        max_email_chars=args.max_email_chars,
        logger=logger,
    )
    if not raw_emails:
        source_name = str(input_path) if input_path is not None else args.hf_dataset
        raise ValueError(f"No emails found for source {source_name}")

    min_required = (
        args.val_size
        + args.test_size
        + args.canary_source_pool_size
        + 1
    )
    if len(raw_emails) < min_required:
        raise ValueError(
            f"Need at least {min_required:,} emails, found {len(raw_emails):,}"
        )

    random.shuffle(raw_emails)

    val_emails = raw_emails[:args.val_size]
    test_emails = raw_emails[args.val_size:args.val_size + args.test_size]
    remaining = raw_emails[args.val_size + args.test_size:]

    canary_source_pool = remaining[:args.canary_source_pool_size]
    pool_emails = remaining[args.canary_source_pool_size:]

    logger.info(
        "Split emails into train=%s val=%s test=%s heldout_prefix_pool=%s",
        f"{len(pool_emails):,}",
        f"{len(val_emails):,}",
        f"{len(test_emails):,}",
        f"{len(canary_source_pool):,}",
    )

    train_records = [
        build_lm_record(f"enron_train_{idx}", text, "train", "enron")
        for idx, text in enumerate(pool_emails)
    ]
    val_records = [
        build_lm_record(f"enron_val_{idx}", text, "val", "enron")
        for idx, text in enumerate(val_emails)
    ]
    test_records = [
        build_lm_record(f"enron_test_{idx}", text, "test", "enron")
        for idx, text in enumerate(test_emails)
    ]

    member_records = []
    member_unique = []
    reference_records = []
    full_registry = []

    for group_idx, (total_insertions, exposure) in enumerate(member_config):
        distinct_count = total_insertions // exposure
        secret_length = {1: 12, 10: 16, 50: 24}.get(exposure, 16)

        logger.info(
            "Creating member tier %s: %s distinct x %s reps",
            group_idx,
            f"{distinct_count:,}",
            exposure,
        )

        for item_idx in range(distinct_count):
            canary_id = f"canary_member_g{group_idx}_{item_idx}"
            source_text = random.choice(canary_source_pool)
            record, registry = generate_canary_from_dataset_prefix(
                canary_id,
                source_text=source_text,
                secret_length=secret_length,
                min_prefix=args.min_prefix,
                max_prefix=args.max_prefix,
            )

            registry["is_member"] = True
            registry["exposure"] = exposure
            full_registry.append(registry)

            eval_record = build_canary_eval_record(
                record=record,
                secret=registry["secret"],
                split="member_eval",
                exposure=exposure,
            )
            member_unique.append(eval_record)

            for rep in range(exposure):
                train_record = {
                    **record,
                    "id": f"{canary_id}_rep{rep}",
                    "split": "train",
                    "exposure": exposure,
                }
                member_records.append(train_record)

    for item_idx in range(args.canary_reference):
        canary_id = f"canary_reference_{item_idx}"
        source_text = random.choice(canary_source_pool)
        record, registry = generate_canary_from_dataset_prefix(
            canary_id,
            source_text=source_text,
            secret_length=16,
            min_prefix=args.min_prefix,
            max_prefix=args.max_prefix,
        )

        registry["is_member"] = False
        registry["exposure"] = 0
        full_registry.append(registry)

        reference_records.append(
            build_canary_eval_record(
                record=record,
                secret=registry["secret"],
                split="reference",
                exposure=0,
            )
        )

    final_train = train_records + member_records
    aux_train_records: List[dict] = []
    aux_val_records: List[dict] = []
    aux_test_records: List[dict] = []

    if args.aux_train_path:
        aux_train_records = load_jsonl_records(
            Path(args.aux_train_path),
            split="train",
            logger=logger,
            limit=args.aux_train_limit,
        )
    if args.aux_val_path:
        aux_val_records = load_jsonl_records(
            Path(args.aux_val_path),
            split="val",
            logger=logger,
            limit=args.aux_val_limit,
        )
    if args.aux_test_path:
        aux_test_records = load_jsonl_records(
            Path(args.aux_test_path),
            split="test",
            logger=logger,
            limit=args.aux_test_limit,
        )

    final_train.extend(aux_train_records)
    val_records.extend(aux_val_records)
    test_records.extend(aux_test_records)
    random.shuffle(final_train)
    random.shuffle(val_records)
    random.shuffle(test_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(final_train, output_dir / "combined_train.jsonl")
    save_jsonl(val_records, output_dir / "combined_val.jsonl")
    save_jsonl(test_records, output_dir / "combined_test.jsonl")
    save_jsonl(reference_records, output_dir / "attack_eval.jsonl")
    save_jsonl(member_unique, output_dir / "member_canaries.jsonl")

    registry_path = output_dir / "canary_registry.json"
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(full_registry, handle, indent=2)

    logger.info("Saved dataset to %s", output_dir)
    logger.info("  train: %s", f"{len(final_train):,}")
    logger.info("  val: %s", f"{len(val_records):,}")
    logger.info("  test: %s", f"{len(test_records):,}")
    logger.info("  member canaries: %s", f"{len(member_unique):,}")
    logger.info("  reference canaries: %s", f"{len(reference_records):,}")
    if aux_train_records or aux_val_records or aux_test_records:
        logger.info(
            "  mixed auxiliary: train=%s val=%s test=%s",
            f"{len(aux_train_records):,}",
            f"{len(aux_val_records):,}",
            f"{len(aux_test_records):,}",
        )


if __name__ == "__main__":
    main()

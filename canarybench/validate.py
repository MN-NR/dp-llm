"""validate(): structural/statistical checks that MUST run before expensive
training, per canarybench spec item 8/E.

Reuses existing checks rather than rewriting them:
  - metrics.privacy.memorization.validate_canary_splits
        (exact member/reference (input, target) overlap)
  - instruction_recall_dataset.build_instruction_dataset.validate_canary_isolation
        (reserved source rows / reserved secrets leaking into ordinary
        training data -- this is the check that found ZERO contamination
        against the current on-disk instruction_recall_dataset processed
        files; preserved here as an automated assertion)

Deliberately does NOT implement a canary-vs-real-data detectability
classifier (out of scope for this pass, per explicit instruction). New
checks can be added by appending to CHECKS below without changing the
report shape.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from canarybench.inject import InjectionArtifacts, load_canonical_canaries
from canarybench.prompt_format import build_supervised_text
from canarybench.schema import VALID_MEMBER_FREQUENCIES, validate_schema


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    check: str
    message: str


@dataclass
class ValidationReport:
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        errors = [i for i in self.issues if i.severity == "error"]
        if errors:
            lines = "\n".join(f"  - [{i.check}] {i.message}" for i in errors)
            raise ValueError(f"CanaryBench validate() failed with {len(errors)} error(s):\n{lines}")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate(
    artifacts: InjectionArtifacts,
    tokenizer: Optional[Any] = None,
    max_seq_length: int = 512,
    print_examples: int = 3,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    stats: Dict[str, Any] = {}

    member_records, reference_records = load_canonical_canaries(artifacts)
    stats["num_members"] = len(member_records)
    stats["num_references"] = len(reference_records)

    # ── required fields ──────────────────────────────────────────────
    for rec in member_records + reference_records:
        try:
            validate_schema(rec.to_dict())
        except ValueError as exc:
            issues.append(ValidationIssue("error", "required_fields", str(exc)))

    # ── member/reference isolation (exact input/target overlap) ─────
    # Reuses metrics.privacy.memorization.validate_canary_splits unchanged.
    from metrics.privacy.memorization import validate_canary_splits

    split_check = validate_canary_splits(
        [r.to_dict() for r in member_records],
        [r.to_dict() for r in reference_records],
    )
    stats["split_validation"] = split_check
    if split_check["has_exact_overlap"]:
        issues.append(ValidationIssue(
            "error", "member_reference_isolation",
            f"{split_check['exact_overlap_count']} member/reference canaries have identical (input, target)",
        ))

    # ── duplicate secrets across member/reference (value-only, not
    #    input+target) -- a subtler leak than exact-pair overlap: two
    #    DIFFERENT canaries (different source rows) coincidentally sharing
    #    the same secret VALUE. Not fatal on its own, but worth surfacing. ─
    member_secrets = {r.target.strip() for r in member_records}
    reference_secrets = {r.target.strip() for r in reference_records}
    secret_value_overlap = member_secrets & reference_secrets
    stats["member_reference_secret_value_overlap_count"] = len(secret_value_overlap)
    if secret_value_overlap:
        issues.append(ValidationIssue(
            "warning", "duplicate_secret_values",
            f"{len(secret_value_overlap)} secret VALUES (not full input/target pairs) appear in both "
            f"member and reference canaries from different source rows -- e.g. coincidentally identical "
            f"synthetic PII values. Does not by itself mean a member/reference pair is identical.",
        ))

    # ── reference secrets absent from training / no accidental leakage ──
    # Reuses instruction_recall_dataset.build_instruction_dataset.validate_canary_isolation
    # unchanged -- this is the check that found ZERO contamination in the
    # current on-disk dataset.
    from instruction_recall_dataset.build_instruction_dataset import validate_canary_isolation

    train_records = _load_jsonl(artifacts.train_path)
    with open(artifacts.registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    ordinary_records = [r for r in train_records if not r.get("canary")]
    try:
        validate_canary_isolation(ordinary_records, registry)
        stats["canary_isolation"] = "passed (0 overlap)"
    except ValueError as exc:
        issues.append(ValidationIssue("error", "canary_isolation", str(exc)))
        stats["canary_isolation"] = f"FAILED: {exc}"

    # ── member secrets present at expected frequency ─────────────────
    from collections import Counter

    canary_train_records = [r for r in train_records if r.get("task") == "secret_recall" and r.get("split") == "train"]
    train_pair_counts = Counter((r["input"], r["target"]) for r in canary_train_records)
    freq_mismatches = []
    for rec in member_records:
        actual = train_pair_counts.get((rec.input, rec.target), 0)
        if actual != rec.frequency:
            freq_mismatches.append((rec.id, rec.frequency, actual))
    stats["frequency_mismatches"] = len(freq_mismatches)
    if freq_mismatches:
        sample = freq_mismatches[:5]
        issues.append(ValidationIssue(
            "error", "expected_frequency",
            f"{len(freq_mismatches)} member canaries do not appear in training at their registered "
            f"frequency (expected, actual) sample: {sample}",
        ))

    freq_counter = Counter(r.frequency for r in member_records)
    stats["frequency_tier_counts"] = dict(sorted(freq_counter.items()))
    for tier in VALID_MEMBER_FREQUENCIES:
        if freq_counter.get(tier, 0) == 0:
            issues.append(ValidationIssue("warning", "frequency_tiers", f"No member canaries at frequency={tier}"))

    # ── PII type distributions ────────────────────────────────────────
    stats["member_field_type_counts"] = dict(Counter(r.field_type for r in member_records))
    stats["reference_field_type_counts"] = dict(Counter(r.field_type for r in reference_records))

    # ── target-length statistics ──────────────────────────────────────
    import numpy as np

    member_target_lens = [len(r.target) for r in member_records]
    reference_target_lens = [len(r.target) for r in reference_records]
    stats["member_target_char_len"] = {
        "mean": float(np.mean(member_target_lens)), "max": max(member_target_lens), "min": min(member_target_lens),
    }
    stats["reference_target_char_len"] = {
        "mean": float(np.mean(reference_target_lens)), "max": max(reference_target_lens), "min": min(reference_target_lens),
    }

    # ── token-length statistics + truncation risk (needs a tokenizer) ──
    if tokenizer is not None:
        truncated = []
        token_lens = []
        for rec in member_records + reference_records:
            prompt_only, full_text = build_supervised_text(tokenizer, rec.to_dict())
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            token_lens.append(len(full_ids))
            if len(full_ids) > max_seq_length:
                # Canary eval paths truncate at max_seq_length (see
                # metrics/base.py::encode_prompt_and_target, truncation=True) --
                # if the target falls past the truncation point, its loss
                # is never actually measured.
                truncated.append(rec.id)
            elif len(prompt_ids) >= max_seq_length:
                truncated.append(rec.id)
        stats["token_length"] = {
            "mean": float(np.mean(token_lens)) if token_lens else None,
            "max": max(token_lens) if token_lens else None,
        }
        stats["num_canaries_at_truncation_risk"] = len(truncated)
        if truncated:
            issues.append(ValidationIssue(
                "error" if len(truncated) > 0.05 * len(token_lens) else "warning",
                "target_truncation",
                f"{len(truncated)} canaries exceed max_seq_length={max_seq_length} tokens (prompt+target); "
                f"their target span would be truncated away during evaluation. Sample ids: {truncated[:5]}",
            ))
    else:
        issues.append(ValidationIssue(
            "warning", "token_length",
            "No tokenizer provided -- skipped token-length/truncation-risk and target-span checks.",
        ))

    # ── prompt formatting + target span (also print for manual inspection) ─
    print(f"\n=== validate(): {print_examples} formatted example(s) ===")
    for rec in member_records[:print_examples]:
        if tokenizer is not None:
            prompt_only, full_text = build_supervised_text(tokenizer, rec.to_dict())
            prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            target_span = tokenizer.decode(full_ids[len(prompt_ids):], skip_special_tokens=True)
        else:
            prompt_only, full_text = build_supervised_text(_NullTokenizer(), rec.to_dict())
            target_span = full_text[len(prompt_only):]
        print(f"--- {rec.id} (is_member={rec.is_member}, frequency={rec.frequency}, field_type={rec.field_type}) ---")
        print("prompt_only:", repr(prompt_only))
        print("full_text:  ", repr(full_text))
        print("target span:", repr(target_span))

    passed = not any(i.severity == "error" for i in issues)
    return ValidationReport(passed=passed, issues=issues, stats=stats)


class _NullTokenizer:
    """Used only to force the "no chat template" fallback branch of
    build_supervised_text when no real tokenizer was provided to validate()."""
    chat_template = None

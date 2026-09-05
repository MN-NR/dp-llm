"""Mandatory, GPU-free-capable checks run before a tuning candidate."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from canarybench.prompt_format import build_generation_prompt, build_supervised_text
from canarybench.validate import validate as validate_canary_artifacts

from .config import DatasetConfig


@dataclass
class PreflightReport:
    passed: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise RuntimeError("candidate preflight failed:\n  - " + "\n  - ".join(self.errors))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_task_label(task: str, generated: str) -> Any:
    """Parse the label-bearing tasks used by processed_v2 validation."""
    first = str(generated or "").strip().splitlines()[0] if str(generated or "").strip() else ""
    first = first.strip(" `\t\"'.:;!,").lower()
    if task == "classification":
        aliases = {
            "world": "World",
            "sports": "Sports",
            "business": "Business",
            "sci/tech": "Sci/Tech",
            "sci-tech": "Sci/Tech",
            "science/technology": "Sci/Tech",
        }
        return aliases.get(first)
    if task == "nli":
        return first if first in {"entailment", "neutral", "contradiction"} else None
    if task in {"pii_detection", "secret_recall"}:
        return [piece.strip() for piece in re.split(r"[;\n]", generated or "") if piece.strip()]
    return first or None


def _sample_records(path: str, count: int = 3) -> List[Dict[str, Any]]:
    preferred = ("classification", "nli", "pii_detection")
    by_task: Dict[str, Dict[str, Any]] = {}
    fallback: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            fallback.append(record)
            task = record.get("task")
            if task in preferred and task not in by_task:
                by_task[task] = record
            if len(by_task) == len(preferred):
                break
    records = [by_task[task] for task in preferred if task in by_task]
    for record in fallback:
        if len(records) >= count:
            break
        if record not in records:
            records.append(record)
    if len(records) < count:
        raise ValueError(f"need {count} training examples for preflight, found {len(records)}")
    return records[:count]


def _terminator_ids(tokenizer: Any) -> set:
    ids = {getattr(tokenizer, "eos_token_id", None)}
    for token in ("<|eot_id|>", "<end_of_turn>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            ids.add(token_id)
    return {int(value) for value in ids if value is not None}


def run_preflight(
    dataset: DatasetConfig,
    tokenizer: Any,
    model: Any,
    lora_target_modules: Sequence[str],
    model_revision: Optional[str],
    tokenizer_revision: Optional[str],
    max_seq_length: int,
) -> PreflightReport:
    """Run all mandatory checks; callers must call ``raise_if_failed``.

    The function deliberately receives no test path and never opens test data.
    ``model`` may be a CPU model; no CUDA operation is required.
    """
    checks: Dict[str, Any] = {"test_accessed": False}
    errors: List[str] = []

    checks["dataset_version"] = dataset.version
    if dataset.version != "processed_v2" or Path(dataset.root).name != "processed_v2":
        errors.append("dataset must be the processed_v2 directory")
    if not model_revision or not tokenizer_revision:
        errors.append("exact model_revision and tokenizer_revision must be recorded")
    checks["model_revision"] = model_revision
    checks["tokenizer_revision"] = tokenizer_revision

    artifacts = SimpleNamespace(
        train_path=dataset.train_path,
        val_path=dataset.validation_path,
        member_path=dataset.member_path,
        reference_path=dataset.reference_path,
        registry_path=dataset.registry_path,
    )
    try:
        canary_report = validate_canary_artifacts(
            artifacts, tokenizer=tokenizer, max_seq_length=max_seq_length, print_examples=3
        )
        checks["canarybench_validation"] = {
            "passed": canary_report.passed,
            "stats": canary_report.stats,
            "issues": [asdict(issue) for issue in canary_report.issues],
        }
        if not canary_report.passed:
            errors.append("existing CanaryBench structural/isolation validation failed")
    except Exception as exc:
        checks["canarybench_validation"] = {"passed": False, "error": str(exc)}
        errors.append(f"CanaryBench validation raised {type(exc).__name__}: {exc}")

    module_names = [name for name, _ in model.named_modules()]
    missing_targets = [
        target for target in lora_target_modules
        if not any(name == target or name.endswith("." + target) for name in module_names)
    ]
    checks["lora_target_modules"] = {
        "requested": list(lora_target_modules),
        "missing": missing_targets,
    }
    if missing_targets:
        errors.append(f"LoRA target modules not present in model: {missing_targets}")

    records = _sample_records(dataset.train_path)
    tokenization_checks: List[Dict[str, Any]] = []
    terminators = _terminator_ids(tokenizer)
    try:
        from finetune.dataset import MultiTaskDataset

        with tempfile.TemporaryDirectory(prefix="canarybench_preflight_") as temp_dir:
            sample_path = Path(temp_dir) / "samples.jsonl"
            with open(sample_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(dict(record, group=record.get("group", "A"))) + "\n")
            training_dataset = MultiTaskDataset(str(sample_path), tokenizer, max_length=max_seq_length)
            if len(training_dataset) != len(records):
                errors.append("one or more preflight examples were unexpectedly filtered by token length")
            for index, record in enumerate(records[:len(training_dataset)]):
                prompt, full_text = build_supervised_text(tokenizer, record)
                eval_prompt = build_generation_prompt(tokenizer, record["input"])
                encoded_default = tokenizer(full_text, add_special_tokens=True)["input_ids"]
                encoded_raw = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                unk_id = getattr(tokenizer, "unk_token_id", None)
                unk_fraction = (
                    sum(token == unk_id for token in encoded_default) / max(len(encoded_default), 1)
                    if unk_id is not None else 0.0
                )
                bos_id = getattr(tokenizer, "bos_token_id", None)
                bos_count = sum(token == bos_id for token in encoded_default) if bos_id is not None else 0
                raw_bos_count = sum(token == bos_id for token in encoded_raw) if bos_id is not None else 0
                tokenizer_added_bos = bool(
                    bos_id is not None
                    and encoded_default
                    and encoded_default[0] == bos_id
                    and (not encoded_raw or encoded_default != encoded_raw)
                )
                expected_bos_count = raw_bos_count + int(tokenizer_added_bos)
                item = training_dataset[index]
                supervised = item["labels"] != -100
                supervised_terminators = [
                    token_id for token_id in terminators
                    if ((item["input_ids"] == token_id) & supervised & (item["attention_mask"] == 1)).any().item()
                ]
                parsed = parse_task_label(record.get("task", ""), record.get("target", ""))
                row = {
                    "task": record.get("task"),
                    "prompt_match": prompt == eval_prompt,
                    "token_count": len(encoded_default),
                    "unk_fraction": unk_fraction,
                    "bos_count": bos_count,
                    "raw_template_bos_count": raw_bos_count,
                    "expected_bos_count": expected_bos_count,
                    "bos_added_by_tokenizer": encoded_default != encoded_raw,
                    "supervised_terminator_ids": supervised_terminators,
                    "label_parser_ok": bool(parsed),
                }
                tokenization_checks.append(row)
                if not row["prompt_match"]:
                    errors.append(f"train/eval prompt mismatch for task {record.get('task')}")
                if unk_fraction > 0.20:
                    errors.append(f"unexpected unk-heavy tokenization for task {record.get('task')}: {unk_fraction:.1%}")
                if bos_count != expected_bos_count:
                    errors.append(
                        f"unexpected BOS behavior for task {record.get('task')}: "
                        f"observed={bos_count}, expected={expected_bos_count}"
                    )
                if not supervised_terminators:
                    errors.append(f"no supervised EOS/EOT terminator for task {record.get('task')}")
                if not row["label_parser_ok"]:
                    errors.append(f"label parser rejected gold-format output for task {record.get('task')}")
    except Exception as exc:
        errors.append(f"tokenization/formatting preflight raised {type(exc).__name__}: {exc}")
    checks["sample_tokenizations"] = tokenization_checks
    checks["three_sample_tokenizations_checked"] = len(tokenization_checks) == 3
    if len(tokenization_checks) != 3:
        errors.append("exactly three sample tokenizations were not verified")

    return PreflightReport(passed=not errors, checks=checks, errors=errors)


def require_preflight(report: PreflightReport) -> None:
    """Explicit launch gate used by candidate runners and unit tests."""
    report.raise_if_failed()

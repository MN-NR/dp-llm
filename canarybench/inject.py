"""inject_canaries(): thin wrapper around the existing, already-validated
canary-generation pipeline in instruction_recall_dataset/build_instruction_dataset.py.

Does not reimplement dataset generation. It calls build_dataset() (an
in-process-callable extraction of that module's old main() body, unchanged
in behavior) and returns an InjectionArtifacts object with the exact paths
it wrote, plus a manifest (injection_metadata.json) describing what was
generated.

No hidden global default directory: callers must pass output_dir
explicitly. This repo's actual dataset currently lives at
instruction_recall_dataset/dataset/processed/ (compat preset below), but
CanaryBench itself makes no assumption about that path.
"""

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from canarybench.schema import CanaryRecord, from_registry_and_record


@dataclass
class InjectionArtifacts:
    train_path: str
    val_path: str
    test_path: str
    member_path: str
    reference_path: str
    registry_path: str
    manifest_path: str
    metadata: Dict[str, Any]


# Compatibility preset for THIS research repo only -- not a CanaryBench
# default. See canarybench/inject.py module docstring and decision #5:
# "do not make that a general CanaryBench assumption."
EXISTING_REPO_DATASET_DIR = os.path.join(
    _PROJECT_ROOT, "instruction_recall_dataset", "dataset", "processed"
)


def inject_canaries(output_dir: str, seed: int = 42) -> InjectionArtifacts:
    """Build (or rebuild) the dataset + member/reference canaries into
    `output_dir`, using the existing, validated build_instruction_dataset.py
    pipeline (770 members at 1x/10x/50x, 1000 references, AI4Privacy-sourced,
    with existing source/secret exclusion + validate_canary_isolation()).

    output_dir must be passed explicitly -- there is no hidden default.
    """
    from instruction_recall_dataset.build_instruction_dataset import (
        CANARY_MEMBER_CONFIG,
        CANARY_REFERENCE_COUNT,
        build_dataset,
    )

    output_dir = str(output_dir)
    summary = build_dataset(Path(output_dir), seed)

    registry_path = os.path.join(output_dir, "canary_registry.json")
    member_path = os.path.join(output_dir, "member_canaries.jsonl")
    reference_path = os.path.join(output_dir, "attack_eval.jsonl")
    train_path = os.path.join(output_dir, "combined_train.jsonl")
    val_path = os.path.join(output_dir, "combined_val.jsonl")
    test_path = os.path.join(output_dir, "combined_test.jsonl")

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    frequency_counts = Counter(int(item.get("exposure", 0)) for item in registry if item.get("is_member"))
    field_type_counts = Counter(str(item.get("field_type", "unknown")) for item in registry)
    num_inserted_member_records = sum(total for total, _ in CANARY_MEMBER_CONFIG)

    metadata = {
        "seed": seed,
        "num_unique_member_canaries": summary["num_member_canaries"],
        "num_reference_canaries": summary["num_reference_canaries"],
        "frequency_tiers": {str(freq): count for freq, count in sorted(frequency_counts.items())},
        "num_actual_inserted_member_records": num_inserted_member_records,
        "pii_type_counts": dict(sorted(field_type_counts.items())),
        "source_dataset": "ai4privacy/open-pii-masking-500k-ai4privacy",
        "generation_config": {
            "canary_member_config": list(CANARY_MEMBER_CONFIG),
            "canary_reference_count": CANARY_REFERENCE_COUNT,
        },
        "train_task_counts": summary["train_task_counts"],
        "val_task_counts": summary["val_task_counts"],
        "test_task_counts": summary["test_task_counts"],
        "canary_isolation_validated": summary["canary_isolation_validated"],
    }

    manifest_path = os.path.join(output_dir, "injection_metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return InjectionArtifacts(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        member_path=member_path,
        reference_path=reference_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        metadata=metadata,
    )


def load_canonical_canaries(artifacts: InjectionArtifacts, injection_seed: Optional[int] = None):
    """Join member_canaries.jsonl / attack_eval.jsonl with canary_registry.json
    into the canonical CanaryRecord schema (canarybench.schema.CanaryRecord).
    Returns (member_records, reference_records) as lists of CanaryRecord.
    """
    with open(artifacts.registry_path, encoding="utf-8") as f:
        registry = {item["id"]: item for item in json.load(f)}

    def _load_and_join(path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                eval_record = json.loads(line)
                registry_entry = registry[eval_record["id"]]
                records.append(from_registry_and_record(registry_entry, eval_record, injection_seed=injection_seed))
        return records

    member_records = _load_and_join(artifacts.member_path)
    reference_records = _load_and_join(artifacts.reference_path)
    return member_records, reference_records

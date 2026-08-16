"""Standard internal canary schema.

One consistent shape for both member and reference canaries. Reference
canaries use the same schema with is_member=False and frequency=0 (they are
never injected into training, see build_reference_canaries's exposure=0).

This does not replace the existing repo's on-disk formats
(member_canaries.jsonl / attack_eval.jsonl / canary_registry.json, produced
by instruction_recall_dataset/build_instruction_dataset.py) -- it is a join
of the two: the eval-record JSONL files carry "input"/"target"/"task" but
not "is_member"/source-row provenance, while canary_registry.json carries
"is_member"/"exposure"/source provenance but not the exact "input" prompt
string. `from_registry_and_record` below joins them by "id".

Field renames from the existing repo's naming, per explicit instruction:
  exposure      -> frequency        ("exposure" is reserved for the Carlini
                                      exposure metric in metrics/privacy/memorization.py;
                                      reusing it for repetition count would be confusing)
  source_row_id -> source_record_id
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

REQUIRED_FIELDS = (
    "id",
    "input",
    "target",
    "is_member",
    "frequency",
    "field_type",
    "task",
)

# Preserves the existing repo's 1x/10x/50x design (CANARY_MEMBER_CONFIG in
# build_instruction_dataset.py). Reference canaries use 0 (never trained on).
VALID_MEMBER_FREQUENCIES = (1, 10, 50)


@dataclass
class CanaryRecord:
    id: str
    input: str
    target: str
    is_member: bool
    frequency: int
    field_type: str
    task: str
    source_record_id: Optional[str] = None
    injection_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_schema(record: Dict[str, Any]) -> None:
    """Raise ValueError if `record` is missing any required schema field."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Canary record {record.get('id', '<no id>')!r} missing required fields: {missing}")


def from_registry_and_record(
    registry_entry: Dict[str, Any],
    eval_record: Dict[str, Any],
    injection_seed: Optional[int] = None,
) -> CanaryRecord:
    """Build a canonical CanaryRecord by joining one canary_registry.json
    entry with its corresponding member_canaries.jsonl / attack_eval.jsonl
    eval record (matched by "id"), as produced by
    instruction_recall_dataset/build_instruction_dataset.py.
    """
    if registry_entry.get("id") != eval_record.get("id"):
        raise ValueError(
            f"id mismatch joining registry/eval record: "
            f"{registry_entry.get('id')!r} != {eval_record.get('id')!r}"
        )
    return CanaryRecord(
        id=eval_record["id"],
        input=eval_record["input"],
        target=eval_record["target"],
        is_member=bool(registry_entry["is_member"]),
        frequency=int(registry_entry.get("exposure", 0)),
        field_type=str(registry_entry.get("field_type", "")),
        task=eval_record.get("task", "secret_recall"),
        source_record_id=registry_entry.get("source_row_id"),
        injection_seed=injection_seed,
    )

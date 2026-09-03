"""Frozen validation-only utility and deterministic selection rules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config import STANDARD_SELECTION_METRIC


REQUIRED_VALIDATION_METRICS = (
    "ag_news_accuracy",
    "mnli_accuracy",
    "pii_unique_f1",
    "pii_exact_match",
    "invalid_rate",
    "repetition_fraction",
)
FORBIDDEN_SELECTION_TERMS = (
    "memorization",
    "member_loss",
    "reference_loss",
    "gap",
    "wbc",
    "exposure",
    "extraction",
    "test",
)


def validate_selection_policy(selection_metric: str, selection_split: str) -> None:
    if selection_split != "validation":
        raise ValueError("checkpoint/configuration selection must use validation only")
    lowered = selection_metric.lower()
    if selection_metric != STANDARD_SELECTION_METRIC or any(term in lowered for term in FORBIDDEN_SELECTION_TERMS):
        raise ValueError(
            f"paper-policy selection_metric must be {STANDARD_SELECTION_METRIC!r}; "
            "memorization, privacy-audit, and test metrics are forbidden"
        )


def compute_u_val(metrics: Mapping[str, Any]) -> float:
    values = []
    for key in ("ag_news_accuracy", "mnli_accuracy", "pii_unique_f1"):
        if key not in metrics:
            raise ValueError(f"missing validation metric: {key}")
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {value}")
        values.append(value)
    return sum(values) / 3.0


def normalize_validation_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    missing = [key for key in REQUIRED_VALIDATION_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"missing validation metrics: {missing}")
    forbidden = [key for key in metrics if any(term in key.lower() for term in FORBIDDEN_SELECTION_TERMS)]
    if forbidden:
        raise ValueError(f"privacy-audit/test metrics are not accepted in selection records: {forbidden}")
    normalized = {key: float(metrics[key]) for key in REQUIRED_VALIDATION_METRICS}
    for key, value in normalized.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {value}")
    normalized["u_val"] = compute_u_val(normalized)
    return normalized


def _checkpoint_key(record: Mapping[str, Any]) -> tuple:
    metrics = record["validation"]
    return (
        -float(metrics["u_val"]),
        -float(metrics["pii_unique_f1"]),
        float(metrics["invalid_rate"]),
        float(metrics["repetition_fraction"]),
        int(record["epoch"]),
    )


def select_checkpoint(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot select a checkpoint without validation records")
    normalized: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["epoch"] = int(item["epoch"])
        item["validation"] = normalize_validation_metrics(item["validation"])
        normalized.append(item)
    return min(normalized, key=_checkpoint_key)


def validate_privacy_budget(target: float, actual: float, tolerance: float = 1e-3) -> None:
    allowed = float(target) + max(float(tolerance), abs(float(target)) * float(tolerance))
    if float(actual) > allowed:
        raise ValueError(
            f"actual epsilon {actual:.8g} exceeds target {target:.8g} "
            f"beyond tolerance {tolerance:g}"
        )


def select_candidate(candidates: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("status") not in {"evaluated", "selected"}:
            continue
        validations = candidate.get("validation", [])
        if not validations:
            continue
        item = dict(candidate)
        item["best_checkpoint"] = select_checkpoint(validations)
        eligible.append(item)
    if not eligible:
        raise ValueError("no completed candidates with validation-only metrics are available")
    return min(eligible, key=lambda item: _checkpoint_key(item["best_checkpoint"]))


from typing import Any, Dict, Optional


def compounding_privacy_score(baseline_gap: float, finetuned_gap: float) -> Optional[float]:
    if baseline_gap == 0:
        return None
    gap_reduction = baseline_gap - finetuned_gap
    return round(float(gap_reduction / abs(baseline_gap)), 4)


def summarize_privacy_profile(
    exposure_result: Dict[str, Any],
    wbc_result: Optional[Dict[str, Any]] = None,
    extraction_result: Optional[Dict[str, Any]] = None,
    baseline_gap: Optional[float] = None,
) -> Dict[str, Any]:
    finetuned_gap = exposure_result.get("gap")
    return {
        "gap": finetuned_gap,
        "exposure_mean": exposure_result.get("exposure_mean"),
        "wbc_auc": (wbc_result or {}).get("wbc_auc"),
        "pii_extraction_rate": (extraction_result or {}).get("pii_extraction_rate"),
        "compounding_score": (
            compounding_privacy_score(baseline_gap, finetuned_gap)
            if baseline_gap is not None and finetuned_gap is not None
            else None
        ),
    }


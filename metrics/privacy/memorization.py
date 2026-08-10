from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from metrics.base import compute_sequence_loss, compute_token_losses, percentile, safe_mean


def validate_canary_splits(
    member_canaries: Sequence[Dict[str, Any]],
    reference_canaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    member_pairs = [(item.get("input", ""), item.get("target", "")) for item in member_canaries]
    reference_pairs = [(item.get("input", ""), item.get("target", "")) for item in reference_canaries]
    member_unique = set(member_pairs)
    reference_unique = set(reference_pairs)
    exact_overlap = member_unique & reference_unique

    return {
        "member_count": len(member_canaries),
        "reference_count": len(reference_canaries),
        "member_unique_count": len(member_unique),
        "reference_unique_count": len(reference_unique),
        "exact_overlap_count": len(exact_overlap),
        "has_exact_overlap": bool(exact_overlap),
        "member_has_field_type_metadata": any(item.get("field_type") is not None for item in member_canaries),
        "member_has_exposure_metadata": any(item.get("exposure") is not None for item in member_canaries),
    }


def compute_canary_losses(
    model: Any,
    tokenizer: Any,
    canaries: Sequence[Dict[str, Any]],
    device: str,
    max_length: int = 512,
    mask_prompt: bool = True,
) -> List[Dict[str, Any]]:
    losses: List[Dict[str, Any]] = []
    for canary in canaries:
        loss = compute_sequence_loss(
            model=model,
            tokenizer=tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=mask_prompt,
        )
        enriched = dict(canary)
        enriched["loss"] = float(loss)
        losses.append(enriched)
    return losses


def _exposure_for_loss(member_loss: float, reference_losses: Sequence[float]) -> float:
    if not reference_losses:
        return 0.0
    # Rank is 1 for the lowest-loss (most memorized) member; higher rank means less exposure.
    # rank can be len(reference_losses)+1 when member_loss exceeds all reference losses.
    # Exposure is conventionally non-negative, so clip at 0.
    n_refs = len(reference_losses)
    rank = sum(1 for ref_loss in reference_losses if ref_loss < member_loss) + 1
    rank = min(rank, n_refs)
    return max(0.0, float(np.log2(n_refs / rank)))


def compute_canary_exposure_from_losses(
    member_losses: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    member_values = [float(item["loss"]) for item in member_losses]
    reference_values = [float(item["loss"]) for item in reference_losses]
    exposures = [_exposure_for_loss(loss, reference_values) for loss in member_values]

    return {
        "exposure_mean": safe_mean(exposures),
        "exposure_p50": percentile(exposures, 50),
        "exposure_p95": percentile(exposures, 95),
        "exposure_max": float(np.max(exposures)) if exposures else None,
        "member_loss_mean": safe_mean(member_values),
        "reference_loss_mean": safe_mean(reference_values),
        "gap": (safe_mean(reference_values) - safe_mean(member_values))
        if member_values and reference_values
        else None,
        "num_members": len(member_values),
        "num_references": len(reference_values),
        "member_exposures": exposures,
    }


def compute_canary_exposure(
    model: Any,
    tokenizer: Any,
    member_canaries: Sequence[Dict[str, Any]],
    reference_canaries: Sequence[Dict[str, Any]],
    device: str = "cuda",
    max_length: int = 512,
    mask_prompt: bool = True,
) -> Dict[str, Any]:
    split_validation = validate_canary_splits(member_canaries, reference_canaries)
    member_losses = compute_canary_losses(
        model=model,
        tokenizer=tokenizer,
        canaries=member_canaries,
        device=device,
        max_length=max_length,
        mask_prompt=mask_prompt,
    )
    reference_losses = compute_canary_losses(
        model=model,
        tokenizer=tokenizer,
        canaries=reference_canaries,
        device=device,
        max_length=max_length,
        mask_prompt=mask_prompt,
    )
    summary = compute_canary_exposure_from_losses(member_losses, reference_losses)
    summary["member_losses"] = member_losses
    summary["reference_losses"] = reference_losses
    summary["split_validation"] = split_validation
    return summary


def exposure_by_category(
    member_losses_with_meta: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ref_values = [float(item["loss"]) for item in reference_losses]
    grouped: Dict[str, List[float]] = defaultdict(list)
    for item in member_losses_with_meta:
        field_type = item.get("field_type")
        if field_type is None:
            continue
        grouped[str(field_type)].append(_exposure_for_loss(float(item["loss"]), ref_values))

    return {
        field_type: {
            "mean_exposure": safe_mean(values),
            "p95_exposure": percentile(values, 95),
            "count": len(values),
        }
        for field_type, values in grouped.items()
    }


def exposure_by_repetition(
    member_losses_with_meta: Sequence[Dict[str, Any]],
    reference_losses: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    ref_values = [float(item["loss"]) for item in reference_losses]
    grouped: Dict[int, List[float]] = defaultdict(list)
    for item in member_losses_with_meta:
        exposure = item.get("exposure")
        if exposure is None:
            continue
        exposure = int(exposure)
        grouped[exposure].append(_exposure_for_loss(float(item["loss"]), ref_values))

    return {
        f"exposure_{exposure}": float(np.mean(values))
        for exposure, values in sorted(grouped.items())
    }


def memorization_by_frequency(canary_results: Sequence[Dict[str, Any]], reference_mean: float) -> Dict[str, Dict[str, Any]]:
    """Break down loss gap by 1x, 10x, 50x exposure tiers (paper table helper)."""
    tiers: Dict[str, Dict[str, Any]] = {}
    for freq in (1, 10, 50):
        losses = [float(r["loss"]) for r in canary_results if r.get("exposure") == freq and r.get("loss") is not None]
        if not losses:
            continue
        member_mean = float(np.mean(losses))
        tiers[f"{freq}x"] = {
            "member_mean": round(member_mean, 4),
            "gap": round(float(reference_mean - member_mean), 4),
            "n": len(losses),
        }
    return tiers


def memorization_by_pii_type(canary_results: Sequence[Dict[str, Any]], reference_mean: float) -> Dict[str, Dict[str, Any]]:
    """Break down loss gap by PII type (field_type) (paper table helper)."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for record in canary_results:
        field_type = str(record.get("field_type", "unknown"))
        loss = record.get("loss")
        if loss is None:
            continue
        grouped[field_type].append(float(loss))

    return {
        field_type: {
            "member_mean": round(float(np.mean(losses)), 4),
            "gap": round(float(reference_mean - float(np.mean(losses))), 4),
            "n": len(losses),
        }
        for field_type, losses in sorted(grouped.items(), key=lambda kv: kv[0])
        if losses
    }


class CanaryExposureEvaluator:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda", max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def compute_canary_losses(
        self,
        canary_path: str,
        max_samples: Optional[int] = None,
        mask_prompt: bool = True,
    ) -> List[Dict[str, Any]]:
        from metrics.base import load_jsonl

        records = load_jsonl(canary_path, max_samples=max_samples)
        return compute_canary_losses(
            model=self.model,
            tokenizer=self.tokenizer,
            canaries=records,
            device=self.device,
            max_length=self.max_length,
            mask_prompt=mask_prompt,
        )

    def compute_per_token_losses(
        self,
        canary_records: Sequence[Dict[str, Any]],
        max_records: int = 50,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for record in canary_records[:max_records]:
            if "input" in record and "target" in record:
                token_losses = compute_token_losses(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    record=record,
                    device=self.device,
                    max_length=self.max_length,
                    mask_prompt=False,
                )
            elif "token_ids" in record:
                ids = torch.tensor(record["token_ids"], device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(ids).logits[:, :-1, :]
                token_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                ).detach().cpu().numpy()
            else:
                token_losses = np.array([])
            results.append({"id": record.get("id", ""), "token_losses": token_losses.tolist()})
        return results

    def compute_exposure(
        self,
        member_losses: Sequence[Dict[str, Any]],
        reference_losses: Sequence[Dict[str, Any]],
    ) -> Optional[float]:
        summary = compute_canary_exposure_from_losses(member_losses, reference_losses)
        return summary.get("exposure_mean")

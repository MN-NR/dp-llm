#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_member_reference_losses(profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    exposure_block = profile.get("canary_exposure") or {}
    member = exposure_block.get("member_losses") or []
    reference = exposure_block.get("reference_losses") or []
    if not isinstance(member, list) or not isinstance(reference, list):
        raise ValueError("profile_matrix JSON missing canary_exposure.member_losses/reference_losses")
    return member, reference


def _roc_auc(labels: List[int], scores: List[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for index, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += index
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _roc_curve(labels: List[int], scores: List[float]) -> List[Dict[str, float]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    curve = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        curve.append(
            {
                "threshold": float(score),
                "fpr": fp / negatives if negatives else 0.0,
                "tpr": tp / positives if positives else 0.0,
            }
        )
    return curve


def _tpr_at_fpr(curve: List[Dict[str, float]], target_fpr: float) -> float:
    eligible = [point["tpr"] for point in curve if point["fpr"] <= target_fpr]
    return float(max(eligible)) if eligible else 0.0


def _bootstrap_auc_ci(
    labels: List[int],
    scores: List[float],
    n_bootstrap_samples: int = 200,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    n = len(labels_arr)
    aucs: List[float] = []
    for _ in range(n_bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample_labels = labels_arr[indices].tolist()
        if len(set(sample_labels)) < 2:
            continue
        sample_scores = scores_arr[indices].tolist()
        aucs.append(_roc_auc(sample_labels, sample_scores))
    if not aucs:
        return {"auc_ci_lower": 0.5, "auc_ci_upper": 0.5}
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return {
        "auc_ci_lower": float(np.percentile(aucs, lower_q)),
        "auc_ci_upper": float(np.percentile(aucs, upper_q)),
    }


def _exposure_for_loss(member_loss: float, reference_losses: List[float]) -> float:
    if not reference_losses:
        return 0.0
    rank = sum(1 for ref_loss in reference_losses if ref_loss < member_loss) + 1
    return float(np.log2(len(reference_losses) / rank))


def _compute_canary_exposure_from_losses(
    member_losses: List[Dict[str, Any]],
    reference_losses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    member_values = [float(item["loss"]) for item in member_losses if item.get("loss") is not None]
    reference_values = [float(item["loss"]) for item in reference_losses if item.get("loss") is not None]
    exposures = [_exposure_for_loss(loss, reference_values) for loss in member_values]
    return {
        "exposure_mean": float(np.mean(exposures)) if exposures else None,
        "exposure_p50": float(np.percentile(exposures, 50)) if exposures else None,
        "exposure_p95": float(np.percentile(exposures, 95)) if exposures else None,
        "exposure_max": float(np.max(exposures)) if exposures else None,
        "member_loss_mean": float(np.mean(member_values)) if member_values else None,
        "reference_loss_mean": float(np.mean(reference_values)) if reference_values else None,
        "gap": (float(np.mean(reference_values)) - float(np.mean(member_values)))
        if member_values and reference_values
        else None,
        "num_members": len(member_values),
        "num_references": len(reference_values),
    }


def _memorization_by_frequency(canary_results: List[Dict[str, Any]], reference_mean: float) -> Dict[str, Dict[str, Any]]:
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


def _memorization_by_pii_type(canary_results: List[Dict[str, Any]], reference_mean: float) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for r in canary_results:
        ft = str(r.get("field_type", "unknown"))
        loss = r.get("loss")
        if loss is None:
            continue
        grouped.setdefault(ft, []).append(float(loss))
    return {
        ft: {
            "member_mean": round(float(np.mean(losses)), 4),
            "gap": round(float(reference_mean - float(np.mean(losses))), 4),
            "n": len(losses),
        }
        for ft, losses in sorted(grouped.items(), key=lambda kv: kv[0])
        if losses
    }


def _mia_from_losses(member: List[Dict[str, Any]], reference: List[Dict[str, Any]]) -> Dict[str, Any]:
    member_losses = [float(r["loss"]) for r in member if r.get("loss") is not None]
    reference_losses = [float(r["loss"]) for r in reference if r.get("loss") is not None]

    labels = [1] * len(member_losses) + [0] * len(reference_losses)
    scores = [-l for l in member_losses] + [-l for l in reference_losses]
    curve = _roc_curve(labels, scores)
    auc = _roc_auc(labels, scores)
    ci = _bootstrap_auc_ci(labels, scores)
    return {
        "mia_auc": round(float(auc), 4),
        "tpr_at_fpr001": round(float(_tpr_at_fpr(curve, 0.01)), 4),
        **{k: round(float(v), 4) for k, v in ci.items()},
        "n_members": len(member_losses),
        "n_references": len(reference_losses),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to results_testing/profile_matrix/*.json")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    profile = _load_json(args.profile)
    member, reference = _extract_member_reference_losses(profile)

    exposure = _compute_canary_exposure_from_losses(member, reference)
    reference_mean = float(exposure["reference_loss_mean"]) if exposure.get("reference_loss_mean") is not None else float("nan")

    report: Dict[str, Any] = {
        "run": profile.get("run"),
        "model_key": profile.get("model_key"),
        "epsilon": profile.get("epsilon"),
        "gap": exposure.get("gap"),
        "exposure_mean_bits": exposure.get("exposure_mean"),
        "mia": _mia_from_losses(member, reference),
        "memorization_by_frequency": _memorization_by_frequency(member, reference_mean),
        "memorization_by_pii_type": _memorization_by_pii_type(member, reference_mean),
    }

    text = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()

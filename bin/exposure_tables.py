#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_member_reference(profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    block = profile.get("canary_exposure") or {}
    member = block.get("member_losses") or []
    reference = block.get("reference_losses") or []
    if not isinstance(member, list) or not isinstance(reference, list):
        raise ValueError(f"Missing canary_exposure.member_losses/reference_losses in {profile.keys()}")
    return member, reference


def _exposure_for_loss(member_loss: float, reference_losses: List[float]) -> float:
    if not reference_losses:
        return 0.0
    rank = sum(1 for ref_loss in reference_losses if ref_loss < member_loss) + 1
    return float(np.log2(len(reference_losses) / rank))


def compute_canary_exposure_from_losses(
    member_losses: List[Dict[str, Any]],
    reference_losses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    member_values = [float(r["loss"]) for r in member_losses if r.get("loss") is not None]
    reference_values = [float(r["loss"]) for r in reference_losses if r.get("loss") is not None]
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


def exposure_by_repetition(
    member_losses: List[Dict[str, Any]],
    reference_losses: List[Dict[str, Any]],
) -> Dict[str, float]:
    ref_values = [float(r["loss"]) for r in reference_losses if r.get("loss") is not None]
    grouped: Dict[int, List[float]] = {}
    for r in member_losses:
        if r.get("loss") is None:
            continue
        exposure = r.get("exposure")
        if exposure is None:
            continue
        exposure_i = int(exposure)
        grouped.setdefault(exposure_i, []).append(_exposure_for_loss(float(r["loss"]), ref_values))
    return {f"{k}x": float(np.mean(v)) for k, v in sorted(grouped.items()) if v}


def exposure_by_category(
    member_losses: List[Dict[str, Any]],
    reference_losses: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ref_values = [float(r["loss"]) for r in reference_losses if r.get("loss") is not None]
    grouped: Dict[str, List[float]] = {}
    for r in member_losses:
        if r.get("loss") is None:
            continue
        ft = str(r.get("field_type", "unknown"))
        grouped.setdefault(ft, []).append(_exposure_for_loss(float(r["loss"]), ref_values))
    return {
        ft: {"mean_exposure": float(np.mean(vals)), "p95_exposure": float(np.percentile(vals, 95)), "n": len(vals)}
        for ft, vals in sorted(grouped.items(), key=lambda kv: kv[0])
        if vals
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for profile_matrix JSONs")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    args = ap.parse_args()

    import glob

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    rows: List[Dict[str, Any]] = []
    full: Dict[str, Any] = {}
    for path in paths:
        prof = _load_json(path)
        member, ref = _extract_member_reference(prof)
        exposure = compute_canary_exposure_from_losses(member, ref)
        freq = exposure_by_repetition(member, ref)
        by_type = exposure_by_category(member, ref)

        key = os.path.basename(path).replace(".json", "")
        full[key] = {"exposure": exposure, "by_frequency": freq, "by_field_type": by_type}

        rows.append(
            {
                "run": prof.get("run"),
                "model_key": prof.get("model_key"),
                "epsilon": prof.get("epsilon"),
                "file": os.path.basename(path),
                "gap": exposure.get("gap"),
                "exposure_mean": exposure.get("exposure_mean"),
                "exposure_p95": exposure.get("exposure_p95"),
                "exposure_max": exposure.get("exposure_max"),
                "freq_1x": freq.get("1x"),
                "freq_10x": freq.get("10x"),
                "freq_50x": freq.get("50x"),
            }
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()


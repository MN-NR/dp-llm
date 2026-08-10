"""
posthoc_canary_metrics.py — Compute exposure metrics from saved canary losses (no GPU required).

This script reads a finetuning results JSON (dp-llm/results_finetuning/*.json) and computes:
  - Carlini exposure-in-bits summary (mean/p50/p95/max)
  - Exposure breakdowns by `exposure` (1/10/50) and by `field_type` (EMAIL/PHONE/...)

It enriches saved losses with metadata from the processed canary JSONL files:
  - member_canaries.jsonl (member metadata; split=member_eval)
  - attack_eval.jsonl (reference metadata; split=attack_eval)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metrics.base import load_jsonl, save_json
from metrics.privacy.memorization import (
    compute_canary_exposure_from_losses,
    exposure_by_category,
    exposure_by_repetition,
)


def _iter_canary_loss_lists(payload: Any) -> Iterable[List[Dict[str, Any]]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("canary_losses"), list):
            yield payload["canary_losses"]
        for value in payload.values():
            yield from _iter_canary_loss_lists(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_canary_loss_lists(item)


def _extract_latest_canary_losses(results: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    all_lists = list(_iter_canary_loss_lists(results))
    return all_lists[-1] if all_lists else None


def _load_metadata_map(processed_dir: Path) -> Dict[str, Dict[str, Any]]:
    member_path = processed_dir / "member_canaries.jsonl"
    reference_path = processed_dir / "attack_eval.jsonl"
    metadata: Dict[str, Dict[str, Any]] = {}

    if member_path.exists():
        for rec in load_jsonl(str(member_path)):
            rec_id = str(rec.get("id", ""))
            if rec_id:
                metadata[rec_id] = {
                    "field_type": rec.get("field_type"),
                    "exposure": rec.get("exposure"),
                }

    if reference_path.exists():
        for rec in load_jsonl(str(reference_path)):
            rec_id = str(rec.get("id", ""))
            if rec_id and rec_id not in metadata:
                metadata[rec_id] = {
                    "field_type": rec.get("field_type"),
                    "exposure": rec.get("exposure"),
                }

    return metadata


def _split_member_reference(
    canary_losses: List[Dict[str, Any]],
    metadata_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    member_losses: List[Dict[str, Any]] = []
    reference_losses: List[Dict[str, Any]] = []

    for item in canary_losses:
        loss = float(item.get("loss", 0.0))
        rec_id = str(item.get("id", ""))
        meta = metadata_map.get(rec_id, {})
        enriched = {
            "id": rec_id,
            "loss": loss,
            "field_type": meta.get("field_type", item.get("field_type", "unk")),
            "exposure": meta.get("exposure", item.get("exposure", 1)),
        }

        is_member = bool(item.get("canary", False))
        if is_member:
            member_losses.append(enriched)
        else:
            reference_losses.append({"id": rec_id, "loss": loss})

    return member_losses, reference_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True, help="Path to a finetuning results JSON file.")
    parser.add_argument(
        "--processed-dir",
        default="dp-llm/instruction_recall_dataset/dataset/processed",
        help="Directory containing member_canaries.jsonl and attack_eval.jsonl.",
    )
    parser.add_argument("--out", default=None, help="Optional output path (JSON).")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    processed_dir = Path(args.processed_dir)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    canary_losses = _extract_latest_canary_losses(results)
    if not canary_losses:
        raise SystemExit(f"No `canary_losses` found in {results_path}")

    metadata_map = _load_metadata_map(processed_dir)
    member_losses, reference_losses = _split_member_reference(canary_losses, metadata_map)

    exposure_summary = compute_canary_exposure_from_losses(member_losses, reference_losses)
    by_exposure = exposure_by_repetition(member_losses, reference_losses)
    by_field_type = exposure_by_category(member_losses, reference_losses)

    out = {
        "results_json": str(results_path),
        "processed_dir": str(processed_dir),
        "num_member_losses": len(member_losses),
        "num_reference_losses": len(reference_losses),
        "exposure_bits": exposure_summary,
        "exposure_by_exposure": by_exposure,
        "exposure_by_field_type": by_field_type,
    }

    if args.out:
        save_json(args.out, out)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


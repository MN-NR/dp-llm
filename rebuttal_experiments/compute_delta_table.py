#!/usr/bin/env python3
"""Compute finetuned-minus-base deltas for every shared evaluation metric."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE = SCRIPT_DIR / "results/base_model_evaluations.json"
DEFAULT_FINETUNED = REPO_ROOT / "results_testing"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/delta_table.csv"
MODEL_KEYS = {"gemma_2b_it", "vault_gemma", "gemma3_1b", "llama3_2_1b_it"}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def flatten_evaluation(evaluation: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    if isinstance(evaluation.get("avg_loss"), (int, float)):
        flat["avg_loss"] = float(evaluation["avg_loss"])
    for section in ("group_losses", "task_losses"):
        for name, value in (evaluation.get(section) or {}).items():
            if isinstance(value, (int, float)):
                flat[f"{section}.{name}"] = float(value)
    for task, metrics in (evaluation.get("metrics") or {}).items():
        for metric, value in (metrics or {}).items():
            if metric != "n_samples" and isinstance(value, (int, float)):
                flat[f"metrics.{task}.{metric}"] = float(value)
    return flat


def variant_from_checkpoint(checkpoint: object) -> str | None:
    text = str(checkpoint or "")
    match = re.search(r"dp_eps([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return f"dp_eps{float(match.group(1)):g}"
    if "baseline" in text.lower():
        return "baseline"
    return None


def load_base(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text())
    output = {}
    for model_key, model_payload in payload.get("models", {}).items():
        for evaluation in model_payload.get("evaluations", []):
            output[(model_key, evaluation["dataset"])] = flatten_evaluation(evaluation)
    return output


def load_finetuned(directory: Path) -> dict[tuple[str, str, str], tuple[float, Path, dict[str, float]]]:
    selected = {}
    paths = sorted(directory.glob("*_all_clean_eval.json"))
    for path in tqdm(paths, desc="Reading finetuned evaluations", unit="file"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model_key = payload.get("model_key")
        variant = variant_from_checkpoint(payload.get("checkpoint"))
        if model_key not in MODEL_KEYS or variant is None:
            continue
        for evaluation in payload.get("evaluations", []):
            dataset = evaluation.get("dataset")
            if dataset not in {"test", "val"}:
                continue
            key = (model_key, variant, dataset)
            candidate = (path.stat().st_mtime, path, flatten_evaluation(evaluation))
            if key not in selected or candidate[0] > selected[key][0]:
                selected[key] = candidate
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--finetuned-results", type=Path, default=DEFAULT_FINETUNED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"[{timestamp()}] Computing finetuned-minus-base deltas")
    base = load_base(args.base_results)
    finetuned = load_finetuned(args.finetuned_results)
    rows = []
    for (model_key, variant, dataset), (_, source_path, metrics) in sorted(finetuned.items()):
        base_metrics = base.get((model_key, dataset), {})
        for metric in sorted(set(base_metrics) & set(metrics)):
            rows.append(
                {
                    "model": model_key,
                    "variant": variant,
                    "dataset": dataset,
                    "metric": metric,
                    "base_value": base_metrics[metric],
                    "finetuned_value": metrics[metric],
                    "delta_finetuned_minus_base": metrics[metric] - base_metrics[metric],
                    "finetuned_source": str(source_path),
                }
            )
    if not rows:
        raise RuntimeError("No shared base/finetuned metrics found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[{timestamp()}] Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

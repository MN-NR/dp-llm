#!/usr/bin/env python3
"""Compute group-privacy bounds and align them with observed canary gaps."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = REPO_ROOT / "instruction_recall_dataset/dataset/processed/canary_registry.json"
DEFAULT_RESULTS = REPO_ROOT / "results/instruction_recall"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"
DEFAULT_POSTHOC = DEFAULT_OUTPUT
EPSILONS = (3.0, 64.0)
FREQUENCIES = (1, 10, 50)
SEEDS = (42, 123)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def trailing_index(value: object) -> int | None:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else None


def load_exposures(registry_path: Path) -> dict[str, int]:
    records = json.loads(registry_path.read_text())
    lookup: dict[str, int] = {}
    for record in records:
        frequency = record.get("exposure")
        if not record.get("is_member") or frequency not in FREQUENCIES:
            continue
        record_id = str(record["id"])
        lookup[record_id] = int(frequency)
        index = trailing_index(record_id)
        if index is not None:
            lookup.setdefault(str(index), int(frequency))
            lookup.setdefault(f"member_canary_{index}", int(frequency))
            lookup.setdefault(f"canary_member_{index}", int(frequency))
    return lookup


def discover_runs(results_dir: Path) -> dict[tuple[float, int], tuple[Path, dict]]:
    selected: dict[tuple[float, int], tuple[Path, dict]] = {}
    paths = sorted(results_dir.glob("vault_gemma_dp_eps*_seed*.json"))
    for path in tqdm(paths, desc="Loading DP results", unit="file"):
        try:
            payload = json.loads(path.read_text())
            key = (float(payload["target_epsilon"]), int(payload["seed"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if key[0] not in EPSILONS or key[1] not in SEEDS or payload.get("pilot"):
            continue
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[key] = (path, payload)
    return selected


def discover_posthoc(
    posthoc_dir: Path,
) -> dict[tuple[float, int, int], tuple[Path, float, int]]:
    selected: dict[tuple[float, int, int], tuple[Path, float, int]] = {}
    paths = sorted(posthoc_dir.glob("canary_eval_eps*_seed*.json"))
    for path in tqdm(paths, desc="Loading post-hoc results", unit="file"):
        try:
            payload = json.loads(path.read_text())
            epsilon = float(payload["epsilon"])
            seed = int(payload["seed"])
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if epsilon not in EPSILONS or seed not in SEEDS:
            continue
        for raw_frequency, metrics in payload.get("gap_by_frequency", {}).items():
            try:
                frequency = int(raw_frequency)
                gap = float(metrics["gap"])
                count = int(metrics["n_members"])
            except (TypeError, ValueError, KeyError):
                continue
            if frequency not in FREQUENCIES:
                continue
            key = (epsilon, seed, frequency)
            previous = selected.get(key)
            if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
                selected[key] = (path, gap, count)
    return selected


def tier_gaps(payload: dict, exposures: dict[str, int]) -> dict[int, tuple[float, int]]:
    epochs = payload.get("epoch_results") or []
    if not epochs:
        return {}
    final_epoch = epochs[-1]
    member_losses: dict[int, list[float]] = defaultdict(list)
    for item in final_epoch.get("canary_losses") or []:
        item_id = str(item.get("id", ""))
        frequency = exposures.get(item_id)
        if frequency is None:
            index = trailing_index(item_id)
            frequency = exposures.get(str(index)) if index is not None else item.get("exposure")
        if frequency in FREQUENCIES and item.get("loss") is not None:
            member_losses[int(frequency)].append(float(item["loss"]))

    references = [
        float(item["loss"])
        for item in (final_epoch.get("reference_losses") or [])
        if item.get("loss") is not None
    ]
    if not references:
        return {}
    reference_mean = sum(references) / len(references)
    return {
        frequency: (reference_mean - sum(values) / len(values), len(values))
        for frequency, values in member_losses.items()
        if values
    }


def build_table(
    registry_path: Path,
    results_dir: Path,
    posthoc_dir: Path,
) -> pd.DataFrame:
    exposures = load_exposures(registry_path)
    runs = discover_runs(results_dir)
    posthoc = discover_posthoc(posthoc_dir)
    rows = []
    for epsilon in EPSILONS:
        by_tier: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
        for (run_epsilon, seed), (_, payload) in runs.items():
            if run_epsilon != epsilon:
                continue
            for frequency, (gap, count) in tier_gaps(payload, exposures).items():
                by_tier[frequency].append((seed, gap, count))

        for (run_epsilon, seed, frequency), (_, gap, count) in posthoc.items():
            if run_epsilon == epsilon:
                by_tier[frequency].append((seed, gap, count))

        for frequency in FREQUENCIES:
            observations = by_tier.get(frequency, [])
            gaps = [item[1] for item in observations]
            rows.append(
                {
                    "epsilon_finetune": epsilon,
                    "frequency_k": frequency,
                    "effective_epsilon": frequency * epsilon,
                    "empirical_gap_mean": (sum(gaps) / len(gaps)) if gaps else math.nan,
                    "empirical_gap_std": pd.Series(gaps).std(ddof=1) if len(gaps) > 1 else (0.0 if gaps else math.nan),
                    "n_runs": len(gaps),
                    "n_member_canaries": max((item[2] for item in observations), default=0),
                    "n_member_observations": sum(item[2] for item in observations),
                    "seeds": ";".join(str(item[0]) for item in observations),
                }
            )
    return pd.DataFrame(rows)


def save_plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {1: "o", 10: "s", 50: "^"}
    for frequency in FREQUENCIES:
        subset = table[table["frequency_k"] == frequency]
        observed = subset.dropna(subset=["empirical_gap_mean"])
        if not observed.empty:
            ax.errorbar(
                observed["effective_epsilon"],
                observed["empirical_gap_mean"],
                yerr=observed["empirical_gap_std"].fillna(0),
                marker=markers[frequency],
                capsize=4,
                linestyle="none",
                label=f"frequency k={frequency}",
            )
    if table["empirical_gap_mean"].notna().sum() == 0:
        ax.text(0.5, 0.5, "Empirical gaps unavailable until DP runs finish", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_xlabel("Group-privacy upper bound: k × ε")
    ax.set_ylabel("Empirical memorization gap (reference − member loss)")
    ax.set_title("Group privacy bound vs. empirical memorization")
    ax.grid(alpha=0.25)
    if ax.has_data():
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--training-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--posthoc-dir", type=Path, default=DEFAULT_POSTHOC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"[{timestamp()}] Computing group-privacy table")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(args.registry, args.training_results, args.posthoc_dir)
    csv_path = args.output_dir / "group_privacy_table.csv"
    plot_path = args.output_dir / "group_privacy_plot.png"
    table.to_csv(csv_path, index=False)
    save_plot(table, plot_path)
    print(f"[{timestamp()}] Wrote {csv_path}")
    print(f"[{timestamp()}] Wrote {plot_path}")
    missing = table[table["n_runs"] == 0]
    if not missing.empty:
        print("WARNING: Some empirical tier gaps are unavailable; training evaluates only a prefix of member canaries by default.")


if __name__ == "__main__":
    main()

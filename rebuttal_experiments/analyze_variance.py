#!/usr/bin/env python3
"""Summarize two-seed memorization gaps and compare epsilon 3 vs 64."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"
EPSILONS = (3.0, 64.0)
SEEDS = (42, 123)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def memorization_gap(payload: dict) -> float:
    epochs = payload.get("epoch_results") or []
    if not epochs:
        raise ValueError("missing epoch_results")
    final_epoch = epochs[-1]
    return float(final_epoch["avg_reference_loss"]) - float(final_epoch["avg_canary_loss"])


def discover(results_dir: Path) -> dict[tuple[float, int], tuple[Path, float]]:
    selected = {}
    for path in tqdm(sorted(results_dir.glob("vault_gemma_dp_eps*_seed*.json")), desc="Loading seeded runs", unit="file"):
        try:
            payload = json.loads(path.read_text())
            key = (float(payload["target_epsilon"]), int(payload["seed"]))
            gap = memorization_gap(payload)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if key[0] not in EPSILONS or key[1] not in SEEDS or payload.get("pilot"):
            continue
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[key] = (path, gap)
    return selected


def save_plot(gaps: dict[float, list[tuple[int, float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_positions = list(range(len(EPSILONS)))
    means, stds = [], []
    for epsilon in EPSILONS:
        values = [gap for _, gap in gaps[epsilon]]
        means.append(sum(values) / len(values) if values else math.nan)
        stds.append(pd.Series(values).std(ddof=1) if len(values) > 1 else 0.0)
    ax.bar(x_positions, means, yerr=stds, capsize=5, color=["#4C78A8", "#F58518"], alpha=0.75)
    for x, epsilon in zip(x_positions, EPSILONS):
        values = gaps[epsilon]
        offsets = [-0.06, 0.06]
        for index, (seed, gap) in enumerate(values):
            ax.scatter(x + offsets[index % len(offsets)], gap, color="black", s=35, zorder=3)
            ax.annotate(f"seed {seed}", (x + offsets[index % len(offsets)], gap), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.set_xticks(x_positions, [f"ε={epsilon:g}" for epsilon in EPSILONS])
    ax.set_ylabel("Memorization gap (reference − member loss)")
    ax.set_title("VaultGemma DP-LoRA: two-seed variance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"[{timestamp()}] Analyzing multi-seed variance")
    selected = discover(args.training_results)
    missing = [(epsilon, seed) for epsilon in EPSILONS for seed in SEEDS if (epsilon, seed) not in selected]
    if missing:
        missing_text = ", ".join(f"epsilon={epsilon:g}/seed={seed}" for epsilon, seed in missing)
        raise RuntimeError(f"Required runs are incomplete: {missing_text}")

    gaps = {
        epsilon: [(seed, selected[(epsilon, seed)][1]) for seed in SEEDS]
        for epsilon in EPSILONS
    }
    rows = []
    for epsilon in EPSILONS:
        values = [gap for _, gap in gaps[epsilon]]
        rows.append(
            {
                "row_type": "epsilon_summary",
                "epsilon": epsilon,
                "n": len(values),
                "seeds": ";".join(str(seed) for seed, _ in gaps[epsilon]),
                "memorization_gaps": ";".join(f"{value:.8g}" for value in values),
                "mean_gap": sum(values) / len(values),
                "std_gap": pd.Series(values).std(ddof=1),
                "comparison": "",
                "welch_t_statistic": math.nan,
                "welch_p_value": math.nan,
            }
        )

    low = [gap for _, gap in gaps[3.0]]
    high = [gap for _, gap in gaps[64.0]]
    test = ttest_ind(low, high, equal_var=False)
    rows.append(
        {
            "row_type": "welch_t_test",
            "epsilon": math.nan,
            "n": len(low) + len(high),
            "seeds": "42;123 per epsilon",
            "memorization_gaps": "",
            "mean_gap": math.nan,
            "std_gap": math.nan,
            "comparison": "epsilon=3 vs epsilon=64",
            "welch_t_statistic": float(test.statistic),
            "welch_p_value": float(test.pvalue),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "variance_analysis.csv"
    plot_path = args.output_dir / "variance_plot.png"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    save_plot(gaps, plot_path)
    print(f"[{timestamp()}] Wrote {csv_path}")
    print(f"[{timestamp()}] Wrote {plot_path}")


if __name__ == "__main__":
    main()

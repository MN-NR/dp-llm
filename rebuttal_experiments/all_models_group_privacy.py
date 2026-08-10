#!/usr/bin/env python3
"""Compare empirical canary-gap scaling with basic DP group-privacy scaling."""

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
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_PROFILE_DIR = WORKSPACE_ROOT / "results_testing/profile_matrix"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
MODEL_KEYS = (
    "gemma_2b_it",
    "gemma3_1b",
    "llama3_2_1b_it",
    "vault_gemma",
)
FREQUENCIES = (1, 10, 50)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def classify_scaling(
    empirical_scaling: float,
    theoretical_scaling: float,
    tolerance: float,
) -> str:
    if not math.isfinite(empirical_scaling):
        return "undefined"
    ratio = empirical_scaling / theoretical_scaling
    if ratio > 1 + tolerance:
        return "exceeds"
    if ratio < 1 - tolerance:
        return "falls_short"
    return "consistent"


def load_runs(profile_dir: Path) -> list[dict]:
    runs = []
    paths = []
    for model in MODEL_KEYS:
        paths.extend(profile_dir.glob(f"{model}__dp_eps*.json"))
    for path in tqdm(sorted(paths), desc="Loading profile runs", unit="file"):
        try:
            payload = json.loads(path.read_text())
            epsilon = float(payload["epsilon"])
            exposure = payload["canary_exposure"]
            members = exposure["member_losses"]
            reference_mean = float(exposure["reference_loss_mean"])
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not members:
            continue
        runs.append(
            {
                "path": path,
                "model": str(payload.get("model_key") or path.name.split("__", 1)[0]),
                "epsilon": epsilon,
                "members": members,
                "reference_mean": reference_mean,
            }
        )
    return runs


def build_table(profile_dir: Path, tolerance: float) -> pd.DataFrame:
    rows = []
    for run in load_runs(profile_dir):
        grouped: dict[int, list[float]] = defaultdict(list)
        for item in run["members"]:
            try:
                frequency = int(item["exposure"])
                loss = float(item["loss"])
            except (TypeError, ValueError, KeyError):
                continue
            if frequency in FREQUENCIES:
                grouped[frequency].append(loss)

        gaps = {
            frequency: run["reference_mean"] - sum(losses) / len(losses)
            for frequency, losses in grouped.items()
            if losses
        }
        gap_1x = gaps.get(1, math.nan)
        for frequency in FREQUENCIES:
            gap = gaps.get(frequency, math.nan)
            empirical_scaling = (
                gap / gap_1x
                if math.isfinite(gap) and math.isfinite(gap_1x) and gap_1x != 0
                else math.nan
            )
            theoretical_scaling = float(frequency)
            scaling_fraction = (
                empirical_scaling / theoretical_scaling
                if math.isfinite(empirical_scaling)
                else math.nan
            )
            classification = (
                "reference"
                if frequency == 1
                else classify_scaling(empirical_scaling, theoretical_scaling, tolerance)
            )
            rows.append(
                {
                    "model": run["model"],
                    "epsilon_finetune": run["epsilon"],
                    "frequency_k": frequency,
                    "effective_epsilon_k_times_epsilon": frequency * run["epsilon"],
                    "n_member_canaries": len(grouped.get(frequency, [])),
                    "reference_loss_mean": run["reference_mean"],
                    "member_loss_mean": (
                        run["reference_mean"] - gap if math.isfinite(gap) else math.nan
                    ),
                    "memorization_gap": gap,
                    "gap_1x": gap_1x,
                    "empirical_gap_scaling_vs_1x": empirical_scaling,
                    "theoretical_linear_scaling": theoretical_scaling,
                    "empirical_over_theoretical": scaling_fraction,
                    "scaling_assessment": classification,
                    "profile_source": str(run["path"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "epsilon_finetune", "frequency_k"]
    )


def save_plot(table: pd.DataFrame, path: Path) -> None:
    selected = table[
        table["epsilon_finetune"].isin([3.0, 64.0])
        & table["frequency_k"].isin(FREQUENCIES)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_KEYS):
        subset = selected[selected["model"] == model]
        for epsilon, marker in ((3.0, "o"), (64.0, "s")):
            run = subset[subset["epsilon_finetune"] == epsilon]
            axis.plot(
                run["frequency_k"],
                run["empirical_gap_scaling_vs_1x"],
                marker=marker,
                label=f"empirical ε={epsilon:g}",
            )
        axis.plot(
            FREQUENCIES,
            FREQUENCIES,
            color="black",
            linestyle="--",
            label="linear group bound",
        )
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_title(model)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Frequency k")
    axes[1, 1].set_xlabel("Frequency k")
    axes[0, 0].set_ylabel("Gap scaling relative to 1×")
    axes[1, 0].set_ylabel("Gap scaling relative to 1×")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Empirical memorization scaling vs. linear DP group bound")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_summary(table: pd.DataFrame, path: Path, tolerance: float) -> None:
    focus = table[
        table["epsilon_finetune"].isin([3.0, 64.0])
        & table["frequency_k"].isin([10, 50])
    ]
    counts = focus["scaling_assessment"].value_counts().to_dict()
    lines = [
        "# All-model group privacy scaling",
        "",
        f"Generated: {timestamp()}",
        "",
        "The basic group-privacy bound scales epsilon linearly: effective_epsilon = k × epsilon.",
        "For descriptive comparison, empirical scaling is the canary memorization gap at k× divided by the 1× gap.",
        f"`consistent` means empirical/theoretical scaling is within ±{tolerance:.0%}; otherwise it is `exceeds` or `falls_short`.",
        "",
        "Important: DP group privacy bounds privacy loss/odds, not cross-entropy loss gaps. "
        "The scaling assessment is therefore a descriptive diagnostic, not a test of the DP theorem.",
        "",
        "## ε=3 and ε=64 assessment counts (10× and 50×)",
        "",
        f"- falls_short: {counts.get('falls_short', 0)}",
        f"- consistent: {counts.get('consistent', 0)}",
        f"- exceeds: {counts.get('exceeds', 0)}",
        f"- undefined: {counts.get('undefined', 0)}",
        "",
        "## Focus table",
        "",
        "| Model | ε | k | kε | Gap | Empirical scaling | Linear prediction | Empirical/theory | Assessment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in focus.iterrows():
        lines.append(
            f"| {row['model']} | {row['epsilon_finetune']:g} | "
            f"{int(row['frequency_k'])} | "
            f"{row['effective_epsilon_k_times_epsilon']:g} | "
            f"{row['memorization_gap']:.5f} | "
            f"{row['empirical_gap_scaling_vs_1x']:.3f} | "
            f"{row['theoretical_linear_scaling']:.0f} | "
            f"{row['empirical_over_theoretical']:.3f} | "
            f"{row['scaling_assessment']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Relative tolerance around linear scaling for 'consistent' (default: 0.2)",
    )
    args = parser.parse_args()

    print(f"[{timestamp()}] Computing all-model group privacy scaling")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(args.profile_dir, args.tolerance)
    if table.empty:
        raise RuntimeError(f"No compatible DP profile results found in {args.profile_dir}")
    csv_path = args.output_dir / "all_models_group_privacy.csv"
    plot_path = args.output_dir / "all_models_group_privacy.png"
    summary_path = args.output_dir / "all_models_group_privacy.md"
    table.to_csv(csv_path, index=False)
    save_plot(table, plot_path)
    save_summary(table, summary_path, args.tolerance)
    print(f"[{timestamp()}] Wrote {csv_path}")
    print(f"[{timestamp()}] Wrote {plot_path}")
    print(f"[{timestamp()}] Wrote {summary_path}")


if __name__ == "__main__":
    main()

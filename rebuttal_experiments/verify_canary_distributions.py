#!/usr/bin/env python3
"""Verify member/reference canary category and secret-length distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from scipy.stats import chi2_contingency, ks_2samp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = REPO_ROOT / "data_prep/data/audit/canary_registry.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/canary_distribution_report.txt"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_populations(path: Path) -> tuple[list[dict], list[dict]]:
    records = json.loads(path.read_text())
    members, references = [], []
    for record in tqdm(records, desc="Reading canary registry", unit="canary"):
        (members if record.get("is_member") else references).append(record)
    return members, references


def fmt_p(value: float) -> str:
    return f"{value:.6g}"


def example_lines(label: str, records: list[dict], count: int = 5) -> list[str]:
    lines = [f"\n{label} examples ({min(count, len(records))}):"]
    for index, record in enumerate(records[:count], start=1):
        prefix = " ".join(str(record.get("prefix", "")).split())
        lines.extend(
            [
                f"  {index}. id={record.get('id')} category={record.get('category')} "
                f"secret_length={len(str(record.get('secret', '')))} exposure={record.get('exposure', 0)}",
                f"     prefix: {prefix[:240]}",
                f"     secret: {record.get('secret', '')}",
            ]
        )
    return lines


def build_report(registry_path: Path) -> str:
    members, references = load_populations(registry_path)
    member_categories = Counter(str(r.get("category", "unknown")) for r in members)
    reference_categories = Counter(str(r.get("category", "unknown")) for r in references)
    categories = sorted(set(member_categories) | set(reference_categories))
    contingency = [
        [member_categories[category] for category in categories],
        [reference_categories[category] for category in categories],
    ]
    chi2, chi_p, chi_dof, _ = chi2_contingency(contingency)

    member_lengths = [len(str(r.get("secret", ""))) for r in members]
    reference_lengths = [len(str(r.get("secret", ""))) for r in references]
    ks = ks_2samp(member_lengths, reference_lengths, alternative="two-sided", method="auto")

    lines = [
        "CANARY DISTRIBUTION VERIFICATION (NeurIPS rebuttal Ad W4)",
        f"Generated: {timestamp()}",
        f"Registry: {registry_path}",
        "",
        f"Member canaries: {len(members)}",
        f"Reference canaries: {len(references)}",
        "",
        "Category distribution:",
        f"{'category':<16}{'member':>12}{'reference':>12}",
    ]
    for category in categories:
        lines.append(f"{category:<16}{member_categories[category]:>12}{reference_categories[category]:>12}")
    lines.extend(
        [
            "",
            "Chi-square homogeneity test (member vs reference categories):",
            f"  chi2={chi2:.6f}, dof={chi_dof}, p={fmt_p(chi_p)}",
            f"  decision at alpha=0.05: {'different distributions' if chi_p < 0.05 else 'no significant difference'}",
            "",
            "Secret-length distribution (characters):",
            f"  member mean={sum(member_lengths)/len(member_lengths):.3f}, min={min(member_lengths)}, max={max(member_lengths)}",
            f"  reference mean={sum(reference_lengths)/len(reference_lengths):.3f}, min={min(reference_lengths)}, max={max(reference_lengths)}",
            "",
            "Two-sample Kolmogorov-Smirnov test (member vs reference lengths):",
            f"  D={ks.statistic:.6f}, p={fmt_p(ks.pvalue)}",
            f"  decision at alpha=0.05: {'different distributions' if ks.pvalue < 0.05 else 'no significant difference'}",
        ]
    )
    lines.extend(example_lines("Member source", members))
    lines.extend(example_lines("Reference source", references))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"[{timestamp()}] Verifying canary distributions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.registry))
    print(f"[{timestamp()}] Wrote {args.output}")


if __name__ == "__main__":
    main()

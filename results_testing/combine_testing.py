import json
import pandas as pd
import glob
import re
from pathlib import Path

def wilson_ci(p: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    This is a better default than normal approximation for accuracies, and only
    needs (p, n). Returns (low, high) in [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = max(0.0, min(1.0, float(p)))
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    return (max(0.0, center - margin), min(1.0, center + margin))


def extract_metrics(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    model = data["model_key"]
    checkpoint = data.get("checkpoint", "")
    # `checkpoint` paths in the JSON are repo-relative; resolve from this script's directory.
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_exists = False
    if checkpoint and pd.notna(checkpoint):
        checkpoint_exists = (repo_root / str(checkpoint)).exists()

    eps_match = re.search(r"dp_eps([0-9.]+)", checkpoint)
    epsilon = eps_match.group(1) if eps_match else "baseline"

    test_eval = next(e for e in data["evaluations"] if e["dataset"] == "test")
    metrics = test_eval["metrics"]
    pii_metrics = metrics.get("pii_detection", {})
    cls = metrics["classification"]
    nli = metrics["nli"]
    cls_acc = cls["accuracy"]
    nli_acc = nli["accuracy"]
    cls_n = int(cls.get("n_samples", 0) or 0)
    nli_n = int(nli.get("n_samples", 0) or 0)
    cls_ci_low, cls_ci_high = wilson_ci(cls_acc, cls_n) if cls_n else (pd.NA, pd.NA)
    nli_ci_low, nli_ci_high = wilson_ci(nli_acc, nli_n) if nli_n else (pd.NA, pd.NA)

    return {
        "model": model,
        "epsilon": epsilon,
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "avg_loss": test_eval["avg_loss"],

        # Utility
        "classification_acc": cls_acc,
        "classification_n": cls_n,
        "classification_ci95_low": cls_ci_low,
        "classification_ci95_high": cls_ci_high,
        "nli_acc": nli_acc,
        "nli_n": nli_n,
        "nli_ci95_low": nli_ci_low,
        "nli_ci95_high": nli_ci_high,

        # PII Detection (utility)
        "pii_precision": pii_metrics.get("precision", pd.NA),
        "pii_recall": pii_metrics.get("recall", pd.NA),
        "pii_f1": pii_metrics.get("f1", pd.NA),
        "pii_exact_match": pii_metrics.get("exact_match", pd.NA),
    }

files = glob.glob("*.json")

rows = [extract_metrics(f) for f in files]
df = pd.DataFrame(rows).sort_values(["model", "epsilon"])

pd.set_option("display.max_columns", None)

print(df)

df.to_csv("results_table.csv", index=False)

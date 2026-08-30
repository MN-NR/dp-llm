"""Downstream validation accuracy/F1 for each Stage 1 checkpoint, via the
existing finetune/test_model.py (unchanged) run against --dataset val on the
frozen processed_v2 dataset. Also evaluates the raw (no-checkpoint) base
model once, for the "base-model utility" / "improvement over base" columns.

Reads results/hp_search_v1/vault_gemma_eps5.0_stage1_manifest.json for the
9 checkpoint paths, calls finetune.test_model.main() for each (writes its own
per-checkpoint JSON, unchanged behavior), then aggregates classification/nli
accuracy + pii_detection scores into one summary JSON for hyperparameter
selection.
"""

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from finetune.config import repo_path
from finetune.test_model import main as test_model_main

MODEL_KEY = "vault_gemma"
EPSILON = 5.0
DATA_DIR = repo_path("instruction_recall_dataset", "dataset", "processed_v2")
RESULTS_DIR = repo_path("results", "hp_search_v1", "vault_gemma", "eps5.0", "downstream_eval")
MANIFEST_PATH = repo_path("results", "hp_search_v1", f"{MODEL_KEY}_eps{EPSILON}_stage1_manifest.json")
SUMMARY_PATH = repo_path("results", "hp_search_v1", f"{MODEL_KEY}_eps{EPSILON}_downstream_summary.json")


def eval_one(tag, checkpoint_path):
    ckpt_tag = "base" if not checkpoint_path else checkpoint_path.strip("/").replace("/", "__")
    out_path = Path(RESULTS_DIR) / f"{MODEL_KEY}_{ckpt_tag}_val_clean_eval.json"
    print(f"\n{'='*70}\nEvaluating {tag}  (checkpoint={checkpoint_path})\n{'='*70}", flush=True)
    start = time.time()
    test_model_main(
        model_key=MODEL_KEY,
        checkpoint_path=checkpoint_path,
        dataset_type="val",
        device="cuda",
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        save_all_samples=True,
    )
    elapsed = time.time() - start
    with open(out_path) as f:
        result = json.load(f)
    ev = result["evaluations"][0]
    metrics = ev["metrics"] or {}
    pii = metrics.get("pii_detection", {})
    summary = {
        "tag": tag,
        "checkpoint": checkpoint_path,
        "eval_runtime_sec": round(elapsed, 1),
        "avg_val_loss": ev["avg_loss"],
        "classification_accuracy": metrics.get("classification", {}).get("accuracy"),
        "nli_accuracy": metrics.get("nli", {}).get("accuracy"),
        "pii_detection": pii,
        "raw_result_path": str(out_path),
    }
    print(f"[{tag}] cls_acc={summary['classification_accuracy']} nli_acc={summary['nli_accuracy']} "
          f"pii_legacy_f1={pii.get('pii_legacy_f1')} pii_unique_f1={pii.get('pii_unique_f1')} "
          f"dupe_frac={pii.get('repetition_frac_examples_with_dupes')} "
          f"mean_dupes={pii.get('repetition_mean_dupe_count')} "
          f"max_dupes={pii.get('repetition_max_dupe_count')} ({elapsed:.0f}s)", flush=True)
    return summary


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    summaries = []

    # Base model (no fine-tuning) first, for the utility-improvement baseline.
    summaries.append(eval_one("base_model_zero_shot", None))

    for run in manifest["runs"]:
        if run["status"] != "completed":
            print(f"Skipping {run['tag']}: status={run['status']}", flush=True)
            continue
        ckpt = run["results_path"]
        summary = eval_one(run["tag"], ckpt)
        summary["learning_rate"] = run["learning_rate"]
        summary["num_epochs"] = run["num_epochs"]
        summary["best_val_loss_training"] = run["best_val_loss"]
        summary["final_epsilon"] = run["final_epsilon"]
        summaries.append(summary)

        Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summaries, f, indent=2)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nDownstream eval complete. Summary: {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()

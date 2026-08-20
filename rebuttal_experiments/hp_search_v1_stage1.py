"""Stage 1 hyperparameter sweep pilot: vault_gemma, epsilon=5.0.

9 independently-calibrated DP runs (3 learning rates x 3 epoch counts).
Each run gets its own Opacus make_private_with_epsilon() calibration inside
finetune.train_dp.train_dp() (keyed off that run's own num_epochs) -- this
script does not touch that calibration logic, it only supplies a distinct
TrainingConfig per run.

Fixed for all 9 runs, per the confirmed pilot decisions:
  max_grad_norm   = 1.0   (unchanged from finetune/config.py's vault_gemma entry)
  dp_physical_batch_size = 2   (unchanged from finetune/config.py's vault_gemma entry)
  precision       = bfloat16  (hardcoded in finetune/train_dp.py, unchanged)
  lora_dropout    = 0.05  (unchanged -- actual paper-config default, not the
                            0.0 originally assumed)
  lora_r / lora_alpha = 16 / 32  (unchanged paper values)
  effective/logical batch size = 64  (batch_size=2 x grad_accum=32, unchanged)
  dataset         = instruction_recall_dataset/dataset/processed_v2/  (the
                     frozen, collision-free dataset from this session's audit
                     -- NOT vault_gemma's stale config.py default)
  epsilon         = 5.0
  delta           = existing paper value (cfg.delta, unchanged)
  seed            = 42

Swept:
  learning_rate in {0.5, 1.0, 2.0} x base_lr, base_lr = 1e-5 (vault_gemma's
    existing config.py value)
  num_epochs in {3, 5, 8}

Checkpoints and results are written to new directories under
checkpoints/hp_search_v1/ and results/hp_search_v1/ -- historical
checkpoints/results are never touched.

Downstream validation accuracy/F1 is NOT computed here (train_dp() only
tracks val_loss). Run finetune/test_model.py separately against each run's
saved `final` checkpoint to get accuracy/F1 for hyperparameter selection.
"""

import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "finetune"))  # train_dp.py does `from config import ...`

from finetune.config import MODELS, repo_path, require_dataset_paths
from finetune.train_dp import train_dp

MODEL_KEY = "vault_gemma"
EPSILON = 5.0
SEED = 42
BASE_LR = 1e-5

LR_MULTIPLIERS = [0.5, 1.0, 2.0]
EPOCH_COUNTS = [3, 5, 8]

DATASET_DIR = repo_path("instruction_recall_dataset", "dataset", "processed_v2")
SWEEP_ROOT = "hp_search_v1"
MANIFEST_PATH = repo_path("results", SWEEP_ROOT, f"{MODEL_KEY}_eps{EPSILON}_stage1_manifest.json")


def build_cfg(lr: float, epochs: int):
    tag = f"lr{lr:.1e}_ep{epochs}"
    cfg = replace(
        MODELS[MODEL_KEY],
        learning_rate=lr,
        num_epochs=epochs,
        train_path=str(Path(DATASET_DIR) / "combined_train.jsonl"),
        val_path=str(Path(DATASET_DIR) / "combined_val.jsonl"),
        canary_path=str(Path(DATASET_DIR) / "member_canaries.jsonl"),
        reference_path=str(Path(DATASET_DIR) / "attack_eval.jsonl"),
        canary_registry_path=str(Path(DATASET_DIR) / "canary_registry.json"),
        output_base=repo_path("checkpoints", SWEEP_ROOT, MODEL_KEY, f"eps{EPSILON}", tag),
        results_dir=repo_path("results", SWEEP_ROOT, MODEL_KEY, f"eps{EPSILON}"),
    )
    require_dataset_paths(cfg)
    return cfg, tag


def main():
    manifest = {
        "model_key": MODEL_KEY,
        "epsilon": EPSILON,
        "seed": SEED,
        "base_lr": BASE_LR,
        "dataset_dir": DATASET_DIR,
        "runs": [],
    }
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)

    configs = [(round(m * BASE_LR, 10), e) for m in LR_MULTIPLIERS for e in EPOCH_COUNTS]

    for i, (lr, epochs) in enumerate(configs, start=1):
        cfg, tag = build_cfg(lr, epochs)
        print(f"\n{'='*70}\n[{i}/{len(configs)}] Starting {tag}  (lr={lr}, epochs={epochs})\n{'='*70}\n", flush=True)
        run_record = {"tag": tag, "learning_rate": lr, "num_epochs": epochs, "status": "running"}
        manifest["runs"].append(run_record)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        start = time.time()
        try:
            results = train_dp(cfg, epsilon=EPSILON, pilot=False, seed=SEED)
            run_record["status"] = "completed"
            run_record["results_path"] = results.get("output_dir") if isinstance(results, dict) else None
            run_record["best_val_loss"] = results.get("best_val_loss") if isinstance(results, dict) else None
            run_record["final_epsilon"] = results.get("final_epsilon") if isinstance(results, dict) else None
            run_record["train_runtime_hrs"] = round((time.time() - start) / 3600, 3)
            print(f"[{i}/{len(configs)}] DONE {tag} in {run_record['train_runtime_hrs']:.2f}h", flush=True)
        except Exception as e:
            run_record["status"] = "failed"
            run_record["error"] = f"{type(e).__name__}: {e}"
            run_record["traceback"] = traceback.format_exc()
            print(f"[{i}/{len(configs)}] FAILED {tag}: {e}", flush=True)

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nStage 1 sweep complete. Manifest:", MANIFEST_PATH, flush=True)


if __name__ == "__main__":
    main()

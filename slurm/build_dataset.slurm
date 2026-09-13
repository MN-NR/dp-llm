#!/usr/bin/env bash
#SBATCH --job-name=dpllm-builddata
#SBATCH --account=pmiu0248
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Rebuild the processed instruction-recall dataset on OSC, deterministically
# (seed 42), into the exact path finetune/config.py's llama3_2_1b_it entry
# expects: instruction_recall_dataset/dataset/processed/
#
# Pulls ag_news, glue/mnli and ai4privacy/open-pii-masking-500k from the HF
# Hub. Cardinal compute nodes have outbound internet (confirmed on cpu).
#
# Submit:  sbatch slurm/build_dataset.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

OUT_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
mkdir -p "$OUT_DIR"

echo ">> building dataset -> $OUT_DIR (seed 42)"
python instruction_recall_dataset/build_instruction_dataset.py \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo
echo ">> outputs:"
ls -la "$OUT_DIR"

echo
echo ">> verifying the paths finetune/config.py wants all exist:"
python - <<'PY'
import os, sys
from finetune.config import MODELS
cfg = MODELS["llama3_2_1b_it"]
need = {
    "train_path": cfg.train_path,
    "val_path": cfg.val_path,
    "canary_path": cfg.canary_path,
    "reference_path": cfg.reference_path,
    "canary_registry_path": cfg.canary_registry_path,
}
missing = {k: v for k, v in need.items() if not os.path.exists(v)}
for k, v in need.items():
    print(f"  {'OK ' if k not in missing else 'MISS'} {k}: {v}")
sys.exit(1 if missing else 0)
PY

echo ">> dataset build complete."

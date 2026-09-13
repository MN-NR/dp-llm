#!/usr/bin/env bash
#SBATCH --job-name=dpllm-warm-hf-cache
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Populate $HF_HOME with one model's full snapshot (weights + tokenizer +
# configs) BEFORE launching several jobs that would otherwise race to
# download it into the same shared cache and corrupt it (partial writes /
# JSONDecodeError on tokenizer_config.json). Real jobs depend on this with
# --dependency=afterok so they all read a complete cache.
#
#   sbatch -M ascend --export=ALL,WARM_MODEL=google/gemma-3-1b-it slurm/warm_hf_cache.slurm
# ---------------------------------------------------------------------------
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
: "${WARM_MODEL:?set WARM_MODEL= via --export}"

echo ">> warming HF cache ($HF_HOME) for: $WARM_MODEL"
python - "$WARM_MODEL" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
m = sys.argv[1]
p = snapshot_download(
    repo_id=m,
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "tokenizer*"],
)
print("snapshot at:", p)
print("files:", sorted(os.listdir(p)))
# sanity: tokenizer + model must load cleanly from the now-complete cache
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
AutoConfig.from_pretrained(m, token=os.environ.get("HF_TOKEN"))
GenerationConfig.from_pretrained(m, token=os.environ.get("HF_TOKEN"))
AutoTokenizer.from_pretrained(m, token=os.environ.get("HF_TOKEN"))
print("OK: config + generation_config + tokenizer load cleanly")
PY
echo ">> done."

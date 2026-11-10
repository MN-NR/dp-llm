# ---------------------------------------------------------------------------
# Shared runtime environment for every dp-llm Slurm job on OSC Cardinal.
# `source` this from the top of each *.slurm script (after the #SBATCH block).
#
# It does NOT load a system CUDA module: the torch 2.5.1+cu124 wheel bundles
# its own CUDA runtime + NCCL, and loading /apps cuda on top only risks
# library-version skew. The GPU driver (from the node) is all that's needed.
# ---------------------------------------------------------------------------
set -euo pipefail

# Repo root: the sbatch submit dir when under Slurm (robust to $0 pointing at
# the slurmd spool copy), else the parent of this file's directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/env.sh" ]]; then
    export DPLLM_ROOT="$SLURM_SUBMIT_DIR"
else
    export DPLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$DPLLM_ROOT"

# --- Python venv (built by bin/osc_setup_env.sh) --------------------------
if [[ ! -x "$DPLLM_ROOT/.venv/bin/python" ]]; then
    echo "ERROR: $DPLLM_ROOT/.venv missing. Run: bash bin/osc_setup_env.sh" >&2
    exit 1
fi
source "$DPLLM_ROOT/.venv/bin/activate"

# --- Caches: HF on scratch (big, re-downloadable), everything else default -
export HF_HOME="/fs/scratch/PMIU0222/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# --- HF token for the gated meta-llama/Llama-3.2-1B-Instruct --------------
# Create ~/.config/dp-llm/hf_token (chmod 600) containing just the token.
_HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/dp-llm/hf_token}"
if [[ -r "$_HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d ' \n\r\t' < "$_HF_TOKEN_FILE")"
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
else
    echo "WARNING: no HF token at $_HF_TOKEN_FILE -- gated model pulls will fail." >&2
    HF_TOKEN=""
    export HF_TOKEN
fi

# --- Determinism / allocator -------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "== dp-llm env =="
echo "  root      : $DPLLM_ROOT"
echo "  python    : $(which python)"
echo "  HF_HOME   : $HF_HOME"
echo "  HF token  : $([[ -n "$HF_TOKEN" ]] && echo present || echo MISSING)"
echo "  node      : $(hostname)"
echo "  job       : ${SLURM_JOB_ID:-<interactive>}  partition=${SLURM_JOB_PARTITION:-?}"
echo "================"

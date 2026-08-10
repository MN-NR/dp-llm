#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULT_DIR="$REPO_ROOT/results"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[$(date --iso-8601=seconds)] ERROR: HF_TOKEN is not set" >&2
    exit 1
fi

run_one() {
    local epsilon="$1"
    local seed="$2"
    local epsilon_tag="${epsilon}.0"
    local run_tag="dp_eps${epsilon_tag}_seed${seed}"
    local checkpoint_dir="$REPO_ROOT/checkpoints/vault_gemma/$run_tag"
    local log_path="$LOG_DIR/vault_gemma_${run_tag}.log"
    local result_glob="$RESULT_DIR/vault_gemma_${run_tag}_*.json"

    if compgen -G "$result_glob" >/dev/null; then
        echo "[$(date --iso-8601=seconds)] SKIP completed $run_tag"
        return
    fi
    if [[ -e "$checkpoint_dir" ]]; then
        echo "[$(date --iso-8601=seconds)] ERROR: $checkpoint_dir exists without a completed result; refusing to overwrite" >&2
        exit 1
    fi

    echo "[$(date --iso-8601=seconds)] START $run_tag"
    "$PYTHON_BIN" -u "$REPO_ROOT/finetune/train_dp.py" \
        --model vault_gemma \
        --epsilon "$epsilon" \
        --seed "$seed" \
        2>&1 | tee "$log_path"
    echo "[$(date --iso-8601=seconds)] DONE $run_tag"
}

# Essential NeurIPS rebuttal matrix only: one GPU, sequential execution.
run_one 3 42
run_one 3 123
run_one 64 42
run_one 64 123

echo "[$(date --iso-8601=seconds)] All four rebuttal runs completed"

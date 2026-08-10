#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/noora/miniconda3/envs/dp_gpu/bin/python}"
RESULT_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one() {
    local epsilon="$1"
    local seed="$2"
    local checkpoint="$REPO_ROOT/checkpoints/instruction_recall/vault_gemma/dp_eps${epsilon}.0_seed${seed}/best"
    local output="$RESULT_DIR/canary_eval_eps${epsilon}_seed${seed}.json"
    local log="$LOG_DIR/canary_eval_eps${epsilon}_seed${seed}.log"

    if [[ -s "$output" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP existing $output"
        return
    fi
    if [[ ! -f "$checkpoint/adapter_config.json" ]]; then
        echo "[$(date --iso-8601=seconds)] ERROR missing adapter: $checkpoint" >&2
        exit 1
    fi

    echo "[$(date --iso-8601=seconds)] START epsilon=$epsilon seed=$seed"
    "$PYTHON_BIN" -u "$SCRIPT_DIR/eval_missing_canaries.py" \
        --checkpoint "$checkpoint" \
        --base-model google/vaultgemma-1b \
        --output "$output" \
        --frequencies 10 50 \
        --batch-size 4 \
        2>&1 | tee "$log"
    echo "[$(date --iso-8601=seconds)] DONE epsilon=$epsilon seed=$seed"
}

run_one 3 42
run_one 3 123
run_one 64 42
run_one 64 123

"$PYTHON_BIN" "$SCRIPT_DIR/group_privacy_analysis.py"
echo "[$(date --iso-8601=seconds)] Post-hoc evaluation and table refresh complete"

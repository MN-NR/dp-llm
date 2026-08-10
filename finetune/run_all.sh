#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Run all experiments in correct order
#
# Usage:
#   chmod +x finetune/run_all.sh
#
#   # Pilot run first (always do this first)
#   ./finetune/run_all.sh --pilot gemma-2-2b-it
#
#   # Full run for one model
#   ./finetune/run_all.sh gemma-2-2b-it
#
#   # Full run for all models
#   ./finetune/run_all.sh
#
# In tmux:
#   tmux new -s training
#   ./finetune/run_all.sh gemma-2-2b-it 2>&1 | tee "$REPO_ROOT/results/gemma-2-2b-it.log"
#   Ctrl+B then D  → detach
#   tmux attach -t training  → reattach
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set; gated Hugging Face models may fail to load"
fi

# Use the caller's current Python environment.
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using active conda env: $CONDA_DEFAULT_ENV"
else
    echo "WARNING: no active conda env detected; using python from PATH"
fi

# Order matters — smallest model first for debugging
ALL_MODELS=("gemma-2-2b-it" "gemma_2b" "vault_gemma" "llama")
EPSILONS=(8.0 5.0 3.0)   # weakest DP first — easiest to converge

PILOT=false
MODELS=()

# Parse args
for arg in "$@"; do
    if [ "$arg" == "--pilot" ]; then
        PILOT=true
    else
        MODELS+=("$arg")
    fi
done

# Default to all models if none specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
fi

echo "============================================================"
echo "Starting experiments: $(date)"
echo "Models   : ${MODELS[@]}"
echo "Epsilons : ${EPSILONS[@]}"
echo "Pilot    : $PILOT"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do

    echo ""
    echo "============================================================"
    echo "MODEL: $MODEL  $(date)"
    echo "============================================================"

    if [ "$PILOT" = true ]; then
        # ── Pilot run only ───────────────────────────────────────────────────
        echo ""
        echo ">>> PILOT Baseline: $MODEL"
        export CUDA_VISIBLE_DEVICES=0
        python train_baseline.py --model "$MODEL" --pilot

        if [ $? -ne 0 ]; then
            echo "❌ Pilot baseline failed for $MODEL"
            exit 1
        fi

        echo ""
        echo ">>> PILOT DP ε=8.0: $MODEL"
        export CUDA_VISIBLE_DEVICES=0
        python train_dp.py --model "$MODEL" --epsilon 8.0 --pilot

        if [ $? -ne 0 ]; then
            echo "❌ Pilot DP failed for $MODEL"
            exit 1
        fi

        echo ""
        echo "✅ Pilot passed for $MODEL — check $REPO_ROOT/results/ for checklist"
        echo "   Run full training with: ./finetune/run_all.sh $MODEL"

    else
        # ── Full run ─────────────────────────────────────────────────────────
        echo ""
        echo ">>> Baseline: $MODEL  $(date)"
        python train_baseline.py --model "$MODEL"

        if [ $? -ne 0 ]; then
            echo "❌ Baseline failed for $MODEL — skipping DP runs"
            continue
        fi
        echo "✅ Baseline done: $MODEL"

        # DP runs — weakest to strongest
        for EPS in "${EPSILONS[@]}"; do
            echo ""
            echo ">>> DP ε=$EPS: $MODEL  $(date)"
            python train_dp.py --model "$MODEL" --epsilon "$EPS"

            if [ $? -ne 0 ]; then
                echo "❌ DP run failed: $MODEL ε=$EPS"
            else
                echo "✅ DP done: $MODEL ε=$EPS"
            fi
        done

        echo ""
        echo "✅ All runs complete for $MODEL"
        echo "   Summarize with: python $SCRIPT_DIR/summarize_results.py"
    fi

done

echo ""
echo "============================================================"
echo "All done: $(date)"
echo "Results in: $REPO_ROOT/results/"
echo "Checkpoints in: $REPO_ROOT/checkpoints/"
echo "============================================================"

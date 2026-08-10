#!/bin/bash
#
# ============================================================
# run_all_training.sh -- Parallel training on 2x L40S
# ============================================================
# VaultGemma (1B) on GPU 0
# Gemma 2B IT on GPU 1
# Gemma 3 1B runs after VaultGemma completes on GPU 0
#
# Usage:
#   chmod +x run_all_training.sh
#   ./run_all_training.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting training sweep at $(date)"
echo "Logs: $LOG_DIR/"
echo "============================================================"

# -- Phase 1: Baselines (multi-GPU with torchrun) ----------------
# Run baselines first since they use both GPUs and are fast.

echo "[$(date)] Phase 1: Baselines"

echo "[$(date)] VaultGemma baseline..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "$SCRIPT_DIR/finetune/train_baseline.py" --model vault_gemma \
    > "$LOG_DIR/vault_gemma_baseline.log" 2>&1
echo "[$(date)] VaultGemma baseline DONE"

echo "[$(date)] Gemma 2B IT baseline..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "$SCRIPT_DIR/finetune/train_baseline.py" --model gemma_2b_it \
    > "$LOG_DIR/gemma_2b_it_baseline.log" 2>&1
echo "[$(date)] Gemma 2B IT baseline DONE"

echo "[$(date)] Gemma 3 1B baseline..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 "$SCRIPT_DIR/finetune/train_baseline.py" --model gemma3_1b \
    > "$LOG_DIR/gemma3_1b_baseline.log" 2>&1
echo "[$(date)] Gemma 3 1B baseline DONE"

# -- Phase 2: DP runs (single GPU each, parallel) ----------------
# VaultGemma on GPU 0, Gemma 2B IT on GPU 1 -- simultaneously.

echo ""
echo "[$(date)] Phase 2: DP training (parallel)"
echo "  GPU 0: VaultGemma epsilon=8,5,3 then Gemma 3 1B epsilon=8"
echo "  GPU 1: Gemma 2B IT epsilon=8,5,3"

(
    echo "[$(date)] [GPU0] VaultGemma epsilon=8..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/finetune/train_dp.py" --model vault_gemma --epsilon 8.0 \
        > "$LOG_DIR/vault_gemma_dp8.log" 2>&1
    echo "[$(date)] [GPU0] VaultGemma epsilon=8 DONE"

    echo "[$(date)] [GPU0] VaultGemma epsilon=5..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/finetune/train_dp.py" --model vault_gemma --epsilon 5.0 \
        > "$LOG_DIR/vault_gemma_dp5.log" 2>&1
    echo "[$(date)] [GPU0] VaultGemma epsilon=5 DONE"

    echo "[$(date)] [GPU0] VaultGemma epsilon=3..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/finetune/train_dp.py" --model vault_gemma --epsilon 3.0 \
        > "$LOG_DIR/vault_gemma_dp3.log" 2>&1
    echo "[$(date)] [GPU0] VaultGemma epsilon=3 DONE"

    echo "[$(date)] [GPU0] Gemma 3 1B epsilon=8..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/finetune/train_dp.py" --model gemma3_1b --epsilon 8.0 \
        > "$LOG_DIR/gemma3_1b_dp8.log" 2>&1
    echo "[$(date)] [GPU0] Gemma 3 1B epsilon=8 DONE"
) &
PID_GPU0=$!

(
    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=8..."
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/finetune/train_dp.py" --model gemma_2b_it --epsilon 8.0 \
        > "$LOG_DIR/gemma_2b_it_dp8.log" 2>&1
    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=8 DONE"

    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=5..."
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/finetune/train_dp.py" --model gemma_2b_it --epsilon 5.0 \
        > "$LOG_DIR/gemma_2b_it_dp5.log" 2>&1
    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=5 DONE"

    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=3..."
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/finetune/train_dp.py" --model gemma_2b_it --epsilon 3.0 \
        > "$LOG_DIR/gemma_2b_it_dp3.log" 2>&1
    echo "[$(date)] [GPU1] Gemma 2B IT epsilon=3 DONE"
) &
PID_GPU1=$!

echo "[$(date)] Waiting for both GPU pipelines..."
echo "  GPU 0 PID: $PID_GPU0"
echo "  GPU 1 PID: $PID_GPU1"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/vault_gemma_dp8.log"
echo "  tail -f $LOG_DIR/gemma_2b_it_dp8.log"
echo "  nvidia-smi -l 10"

wait "$PID_GPU0"
wait "$PID_GPU1"

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE at $(date)"
echo "============================================================"
echo ""
echo "Total runs completed: 10"
echo "  VaultGemma:   baseline, epsilon=8, epsilon=5, epsilon=3"
echo "  Gemma 2B IT:  baseline, epsilon=8, epsilon=5, epsilon=3"
echo "  Gemma 3 1B:   baseline, epsilon=8"
echo ""
echo "Next steps:"
echo "  python $SCRIPT_DIR/finetune/summarize_results.py"
echo "  # Then run lm-eval-harness on all checkpoints"

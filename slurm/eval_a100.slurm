#!/usr/bin/env bash
#SBATCH --job-name=dpllm-eval-a100
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Post-training evaluation for the completed 20-epoch Llama ε=5 run
# (checkpoints/instruction_recall/llama3_2_1b_it/dp_eps5.0_seed42_batch128_epochs20_2gpu/).
# Single GPU -- this is inference only, no DDP/NCCL, so the Ascend P2P issue
# (see train_paper_a100.slurm) doesn't apply here.
#
#   Phase 1: slurm/select_checkpoint.py -- validation utility ONLY across
#            all 20 epoch checkpoints (AG News / MNLI greedy-gen accuracy +
#            invalid rate, PII unique-F1/exact-match/repetition). No
#            memorization/exposure metric is computed in this phase.
#   Phase 2: slurm/privacy_audit.py -- full privacy audit on the ONE
#            selected checkpoint only (actual epsilon, member/reference
#            loss gap + bootstrap CI, WBC AUC + CI, exposure, 1x/10x/50x
#            breakdown, targeted greedy extraction), plus a full-val
#            utility re-confirmation.
#
#   sbatch -M ascend slurm/eval_a100.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/llama3_2_1b_it/dp_eps5.0_seed42_batch128_epochs20_2gpu"
DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

nvidia-smi -L

echo
echo "### PHASE 1: checkpoint selection (validation utility only) ###"
SELECT_OUT="$OUT_DIR/checkpoint_selection.json"
python slurm/select_checkpoint.py \
    --ckpt-root "$CKPT_ROOT" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --epochs 20 \
    --max-per-task 300 \
    --device cuda \
    --out "$SELECT_OUT" | tee "$OUT_DIR/select_checkpoint.log"

SELECTED_EPOCH=$(grep -oE 'SELECTED_EPOCH=[0-9]+' "$OUT_DIR/select_checkpoint.log" | tail -1 | cut -d= -f2)
if [[ -z "$SELECTED_EPOCH" ]]; then
    echo "ERROR: could not parse SELECTED_EPOCH from select_checkpoint.py output." >&2
    exit 3
fi
echo ">> selected epoch: $SELECTED_EPOCH"

echo
echo "### PHASE 2: privacy audit on epoch_${SELECTED_EPOCH} only ###"
python slurm/privacy_audit.py \
    --ckpt-root "$CKPT_ROOT" \
    --epoch "$SELECTED_EPOCH" \
    --member-path "$DATA_DIR/member_canaries.jsonl" \
    --reference-path "$DATA_DIR/attack_eval.jsonl" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --device cuda \
    --out "$OUT_DIR/privacy_audit_epoch_${SELECTED_EPOCH}.json"

echo
echo ">> Done. See:"
echo "     $SELECT_OUT"
echo "     $OUT_DIR/privacy_audit_epoch_${SELECTED_EPOCH}.json"

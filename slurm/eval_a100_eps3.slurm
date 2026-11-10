#!/usr/bin/env bash
#SBATCH --job-name=dpllm-eval-a100-eps3
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
# Same as eval_a100.slurm but for the epsilon=3.0 run
# (checkpoints/instruction_recall/llama3_2_1b_it/dp_eps3.0_seed42_batch128_epochs20_2gpu/).
# Output filenames are eps-tagged so they don't collide with the eps=5 eval's
# (which used untagged names -- checkpoint_selection.json etc).
#
# Submit with a Slurm dependency on the eps=3 training job so this only
# starts after that training completes successfully:
#   sbatch -M ascend --dependency=afterok:<train_job_id> slurm/eval_a100_eps3.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/llama3_2_1b_it/dp_eps3.0_seed42_batch128_epochs20_2gpu"
DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

nvidia-smi -L

echo
echo "### PHASE 1: checkpoint selection (validation utility only) -- eps=3.0 ###"
SELECT_OUT="$OUT_DIR/checkpoint_selection_eps3.json"
python slurm/select_checkpoint.py \
    --ckpt-root "$CKPT_ROOT" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --epochs 20 \
    --max-per-task 300 \
    --device cuda \
    --out "$SELECT_OUT" | tee "$OUT_DIR/select_checkpoint_eps3.log"

SELECTED_EPOCH=$(grep -oE 'SELECTED_EPOCH=[0-9]+' "$OUT_DIR/select_checkpoint_eps3.log" | tail -1 | cut -d= -f2)
if [[ -z "$SELECTED_EPOCH" ]]; then
    echo "ERROR: could not parse SELECTED_EPOCH from select_checkpoint.py output." >&2
    exit 3
fi
echo ">> selected epoch: $SELECTED_EPOCH"

echo
echo "### PHASE 2: privacy audit on epoch_${SELECTED_EPOCH} only -- eps=3.0 ###"
python slurm/privacy_audit.py \
    --ckpt-root "$CKPT_ROOT" \
    --epoch "$SELECTED_EPOCH" \
    --member-path "$DATA_DIR/member_canaries.jsonl" \
    --reference-path "$DATA_DIR/attack_eval.jsonl" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --device cuda \
    --out "$OUT_DIR/privacy_audit_eps3_epoch_${SELECTED_EPOCH}.json"

echo
echo ">> Done. See:"
echo "     $SELECT_OUT"
echo "     $OUT_DIR/privacy_audit_eps3_epoch_${SELECTED_EPOCH}.json"

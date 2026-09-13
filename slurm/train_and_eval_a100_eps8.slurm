#!/usr/bin/env bash
#SBATCH --job-name=dpllm-llama-eps8-a100
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Train AND evaluate in one job: 20-epoch DPDDP training (epsilon=8.0, same
# batch/LR/seed/LoRA config as eps=3/eps=5, Opacus calibrates a smaller
# noise multiplier for this looser budget) immediately followed by
# checkpoint selection + privacy audit, all on the same node/allocation --
# no separate eval job, no second wait in the Ascend queue.
#
# Training uses both GPUs (DPDDP); the eval phase only needs one and just
# uses the default device (cuda:0) -- the second GPU sits idle during eval,
# a minor inefficiency traded for not having to coordinate two jobs.
#
# Writes to checkpoints/instruction_recall/llama3_2_1b_it/dp_eps8.0_seed42_batch128_epochs20_2gpu/
# and results/instruction_recall/{checkpoint_selection_eps8,privacy_audit_eps8_epoch_N}.json
# -- distinct from the eps=3 and eps=5 runs, no collision.
#
#   sbatch -M ascend slurm/train_and_eval_a100_eps8.slurm
#
# Same NCCL_P2P_DISABLE=1 / pb=16 / --cpus-per-task=32-to-srun fixes as the
# other two runs -- see train_paper_a100.slurm's comments for why.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export NCCL_P2P_DISABLE=1

GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = 128 global logical batch
GRAD_ACCUM=8
PHYS_BATCH=16
LR=2e-5
EPS=8.0
SEED=42
EPOCHS=20

echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py
echo

nvidia-smi
echo ">> global batch $GLOBAL_BATCH  (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  eps $EPS  epochs $EPOCHS  seed $SEED"

echo
echo "### TRAINING ###"
srun --cpus-per-task=32 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    finetune/train_dp_ddp.py \
        --model llama3_2_1b_it \
        --epsilon "$EPS" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --physical-batch-size "$PHYS_BATCH" \
        --learning-rate "$LR" \
        --epochs "$EPOCHS" \
        --save-every-epoch \
        --hf_token "$HF_TOKEN"
TRAIN_RC=$?
echo ">> training finished (rc=$TRAIN_RC)."
if [[ $TRAIN_RC -ne 0 ]]; then
    echo "ERROR: training failed, skipping evaluation." >&2
    exit $TRAIN_RC
fi

CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/llama3_2_1b_it/dp_eps${EPS}_seed${SEED}_batch${GLOBAL_BATCH}_epochs${EPOCHS}_2gpu"
DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

echo
echo "### PHASE 1: checkpoint selection (validation utility only) -- eps=$EPS ###"
SELECT_OUT="$OUT_DIR/checkpoint_selection_eps8.json"
python slurm/select_checkpoint.py \
    --ckpt-root "$CKPT_ROOT" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --epochs "$EPOCHS" \
    --max-per-task 300 \
    --device cuda \
    --out "$SELECT_OUT" | tee "$OUT_DIR/select_checkpoint_eps8.log"

SELECTED_EPOCH=$(grep -oE 'SELECTED_EPOCH=[0-9]+' "$OUT_DIR/select_checkpoint_eps8.log" | tail -1 | cut -d= -f2)
if [[ -z "$SELECTED_EPOCH" ]]; then
    echo "ERROR: could not parse SELECTED_EPOCH from select_checkpoint.py output." >&2
    exit 3
fi
echo ">> selected epoch: $SELECTED_EPOCH"

echo
echo "### PHASE 2: privacy audit on epoch_${SELECTED_EPOCH} only -- eps=$EPS ###"
python slurm/privacy_audit.py \
    --ckpt-root "$CKPT_ROOT" \
    --epoch "$SELECTED_EPOCH" \
    --member-path "$DATA_DIR/member_canaries.jsonl" \
    --reference-path "$DATA_DIR/attack_eval.jsonl" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --device cuda \
    --out "$OUT_DIR/privacy_audit_eps8_epoch_${SELECTED_EPOCH}.json"

echo
echo ">> Done. See:"
echo "     $SELECT_OUT"
echo "     $OUT_DIR/privacy_audit_eps8_epoch_${SELECTED_EPOCH}.json"

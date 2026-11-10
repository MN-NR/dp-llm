#!/usr/bin/env bash
#SBATCH --job-name=dpllm-vaultgemma-eps8-seed43
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
# VaultGemma DP-LoRA, epsilon=8.0, SEED 43 -- train + checkpoint selection +
# privacy audit in one job (combined pattern). Independent of the seed-42
# LS04 VaultGemma runs and of the Llama runs.
#
# Held fixed to the frozen VaultGemma pipeline (only the seed changes):
#   dataset            = instruction_recall_dataset/dataset/processed (seed-42 build)
#   global batch       = 128  (--batch-size 16 --grad-accum 8)
#   LoRA               = config.py vault_gemma target_modules (7 proj), r/alpha default
#   clip (max_grad_norm) = 1.0  (config.py vault_gemma)
#   LR                 = 2e-5  (the selected LR policy; config.py's 1e-5 is stale)
#   EOS / prompt       = build_supervised_text canonical pipeline (Bug 3 fix in place)
#   epochs             = 20, --save-every-epoch  (same protocol as the Llama sweep)
#   max_seq_length     = 128  (config.py vault_gemma; NOTE: Llama runs used 256 --
#                              this is the VaultGemma pipeline's own value, kept fixed)
# Noise multiplier is recalibrated by Opacus for THIS epsilon (nothing to set).
#
# Physical batch 16 and the Ascend fixes (NCCL_P2P_DISABLE=1,
# --cpus-per-task to srun) as in the Llama A100 runs.
#
#   sbatch -M ascend slurm/train_and_eval_a100_vaultgemma_eps8_seed43.slurm
#
# REQUIRES: HF access to google/vaultgemma-1b granted for the account (as of
# 2026-09-05 it was "awaiting review"). The preflight below fails fast if not.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export NCCL_P2P_DISABLE=1
export DPLLM_MODEL_KEY=vault_gemma

MODEL=vault_gemma
EPS=8.0
SEED=43
GLOBAL_BATCH=128
BATCH_SIZE=16
GRAD_ACCUM=8
PHYS_BATCH=16
LR=2e-5
EPOCHS=20

DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

echo "### PREFLIGHT: HF gated-repo content access (vault_gemma) ###"
python slurm/check_hf_gated.py

nvidia-smi
echo ">> $MODEL  eps $EPS  seed $SEED  global batch $GLOBAL_BATCH (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  epochs $EPOCHS"

echo
echo "### TRAINING ###"
srun --cpus-per-task=32 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    finetune/train_dp_ddp.py \
        --model "$MODEL" \
        --epsilon "$EPS" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --physical-batch-size "$PHYS_BATCH" \
        --learning-rate "$LR" \
        --epochs "$EPOCHS" \
        --save-every-epoch \
        --hf_token "$HF_TOKEN" \
        --train-path          "$DATA_DIR/combined_train.jsonl" \
        --val-path            "$DATA_DIR/combined_val.jsonl" \
        --canary-path         "$DATA_DIR/member_canaries.jsonl" \
        --reference-path      "$DATA_DIR/attack_eval.jsonl" \
        --canary-registry-path "$DATA_DIR/canary_registry.json" \
        --output-base         "$DPLLM_ROOT/checkpoints/instruction_recall" \
        --results-dir         "$OUT_DIR"
TRAIN_RC=$?
echo ">> training finished (rc=$TRAIN_RC)."
[[ $TRAIN_RC -ne 0 ]] && { echo "ERROR: training failed, skipping eval." >&2; exit $TRAIN_RC; }

CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/$MODEL/dp_eps${EPS}_seed${SEED}_batch${GLOBAL_BATCH}_epochs${EPOCHS}_2gpu"
TAG="vaultgemma_eps8_seed43"

echo
echo "### PHASE 1: checkpoint selection (validation utility only) ###"
python slurm/select_checkpoint.py \
    --model-key "$MODEL" \
    --ckpt-root "$CKPT_ROOT" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --epochs "$EPOCHS" --max-per-task 300 --seed 42 --device cuda \
    --out "$OUT_DIR/checkpoint_selection_${TAG}.json" | tee "$OUT_DIR/select_checkpoint_${TAG}.log"

SELECTED_EPOCH=$(grep -oE 'SELECTED_EPOCH=[0-9]+' "$OUT_DIR/select_checkpoint_${TAG}.log" | tail -1 | cut -d= -f2)
[[ -z "$SELECTED_EPOCH" ]] && { echo "ERROR: no SELECTED_EPOCH parsed." >&2; exit 3; }
echo ">> selected epoch: $SELECTED_EPOCH"

echo
echo "### PHASE 2: privacy audit on epoch_${SELECTED_EPOCH} ###"
python slurm/privacy_audit.py \
    --model-key "$MODEL" \
    --ckpt-root "$CKPT_ROOT" \
    --epoch "$SELECTED_EPOCH" \
    --member-path "$DATA_DIR/member_canaries.jsonl" \
    --reference-path "$DATA_DIR/attack_eval.jsonl" \
    --val-path "$DATA_DIR/combined_val.jsonl" \
    --device cuda \
    --out "$OUT_DIR/privacy_audit_${TAG}_epoch_${SELECTED_EPOCH}.json"

echo
echo ">> Done. See:"
echo "     $OUT_DIR/checkpoint_selection_${TAG}.json"
echo "     $OUT_DIR/privacy_audit_${TAG}_epoch_${SELECTED_EPOCH}.json"

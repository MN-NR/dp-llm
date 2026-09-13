#!/usr/bin/env bash
#SBATCH --job-name=dpllm-generic
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
# GENERIC DP-LoRA combined job: train (2-GPU Opacus DDP) + checkpoint
# selection (U_val ONLY) + full CanaryBench privacy audit, in one job.
#
# Parametrised via --export -- submit like:
#   sbatch -M ascend -J dpllm-gemma3-eps3-seed42 \
#     --export=ALL,MODEL=gemma3_1b,EPS=3.0,SEED=42,TAG=gemma3_eps3_seed42 \
#     slurm/train_and_eval_a100_generic.slurm
#
# Required env: MODEL (config.py MODELS key), EPS (target epsilon, e.g. 3.0),
#               SEED (int), TAG (short slug for the result filenames).
#
# FROZEN corrected pipeline -- held fixed for every job, only MODEL/EPS/SEED
# vary:
#   dataset            = instruction_recall_dataset/dataset/processed (seed-42 build)
#   global batch       = 128   (--batch-size 16 --grad-accum 8)
#   max_grad_norm      = 1.0   (config.py base default)
#   LoRA               = config.py per-model target_modules (7 proj), r16/alpha32,
#                        dropout forced to 0 by train_dp_ddp.py (Opacus)
#   LR                 = 2e-5  (frozen; NOT retuned for new seeds)
#   EOS / prompt       = build_supervised_text canonical pipeline (Bug 2+3 fixes)
#   epochs             = 20, --save-every-epoch
#   max_seq_length     = config.py per-model (256; vault_gemma 128)
# Noise multiplier is recalibrated by Opacus for THIS epsilon (nothing to set).
# Physical batch 16 + Ascend NCCL_P2P_DISABLE=1 as in the Llama/VaultGemma A100 runs.
# ---------------------------------------------------------------------------
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export NCCL_P2P_DISABLE=1

: "${MODEL:?set MODEL= via --export}"
: "${EPS:?set EPS= via --export}"
: "${SEED:?set SEED= via --export}"
: "${TAG:?set TAG= via --export}"
export DPLLM_MODEL_KEY="$MODEL"

GLOBAL_BATCH=128
BATCH_SIZE=16
GRAD_ACCUM=8
PHYS_BATCH="${PHYS_BATCH:-16}"   # per-hardware VRAM knob; drop for bigger models on A100-40GB
LR="${LR:-2e-5}"                 # frozen protocol LR; override only for a sensitivity run
EPOCHS=20

DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

echo "### PREFLIGHT: HF gated-repo content access ($MODEL) ###"
python slurm/check_hf_gated.py

echo "### PREFLIGHT: EOS-supervised formatting ($MODEL) ###"
python slurm/verify_eos_supervision.py --model-key "$MODEL" --hf-token "$HF_TOKEN" \
    --train-path "$DATA_DIR/combined_train.jsonl" \
    || { echo "ERROR: EOS-supervision verification failed for $MODEL." >&2; exit 4; }

nvidia-smi
echo ">> $MODEL  eps $EPS  seed $SEED  global batch $GLOBAL_BATCH (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  epochs $EPOCHS  tag $TAG"

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

echo
echo "### PHASE 1: checkpoint selection (validation utility ONLY) ###"
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
echo "### PHASE 2: full CanaryBench privacy audit on epoch_${SELECTED_EPOCH} ###"
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
echo ">> Done ($TAG). See:"
echo "     $OUT_DIR/checkpoint_selection_${TAG}.json"
echo "     $OUT_DIR/privacy_audit_${TAG}_epoch_${SELECTED_EPOCH}.json"

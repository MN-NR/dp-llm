#!/usr/bin/env bash
#SBATCH --job-name=dpllm-vaultgemma-baseline-nodp-seed42
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# VaultGemma-1B, NON-DP (epsilon = inf) LoRA baseline, seed 42, corrected
# EOS-supervised pipeline. Pairs with the DP VaultGemma sweep as the
# privacy-utility reference point.  train + checkpoint selection (U_val
# ONLY) + full CanaryBench audit in one job.
#
#   sbatch -M ascend slurm/train_and_eval_a100_baseline_vaultgemma.slurm
#
# Held fixed to the frozen corrected pipeline (matches the DP runs except
# for the DP mechanism itself):
#   dataset          = instruction_recall_dataset/dataset/processed (seed-42 build)
#   global batch     = 128  (--batch-size 16 --grad-accum 8; plain grad-accum,
#                            no Poisson sampling / privacy accounting)
#   LoRA             = vault_gemma target_modules (7 proj), r16/alpha32,
#                      dropout 0.0  (--lora-dropout 0.0, matches the DP path
#                      which forces dropout=0 for Opacus)
#   LR               = 2e-5
#   EOS / prompt      = build_supervised_text canonical pipeline (shared dataset.py)
#   epochs           = 20, --save-every-epoch  (epoch_N/ adapters; HF
#                      best-by-eval_loss selection disabled -- selection is U_val)
#   max_seq_length   = 128  (config.py vault_gemma value)
# Single GPU: VaultGemma-1B fits easily (~3-4 GB in the DP runs); no
# multi-GPU path is needed or wanted for a non-DP HF Trainer run.
# ---------------------------------------------------------------------------
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export DPLLM_MODEL_KEY=vault_gemma

MODEL=vault_gemma
SEED=42
EPOCHS=20
BATCH_SIZE=16
GRAD_ACCUM=8
LR=2e-5

DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
mkdir -p "$OUT_DIR"

echo "### PREFLIGHT: HF gated-repo content access (vault_gemma) ###"
python slurm/check_hf_gated.py

echo "### PREFLIGHT: EOS-supervised formatting (vault_gemma) ###"
python slurm/verify_eos_supervision.py --model-key vault_gemma --hf-token "$HF_TOKEN" \
    --train-path "$DATA_DIR/combined_train.jsonl" \
    || { echo "ERROR: EOS-supervision verification failed for vault_gemma." >&2; exit 4; }

nvidia-smi
echo ">> $MODEL  NON-DP baseline  seed $SEED  global batch 128 (bs $BATCH_SIZE x ga $GRAD_ACCUM)  LR $LR  epochs $EPOCHS"

echo
echo "### TRAINING (non-DP) ###"
python finetune/train_baseline.py \
    --model "$MODEL" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LR" \
    --epochs "$EPOCHS" \
    --lora-dropout 0.0 \
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

CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/$MODEL/baseline_seed${SEED}_epochs${EPOCHS}"
TAG="vaultgemma_baseline_nodp_seed42"

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
echo ">> Done ($TAG). actual_epsilon will be null (non-DP). See:"
echo "     $OUT_DIR/checkpoint_selection_${TAG}.json"
echo "     $OUT_DIR/privacy_audit_${TAG}_epoch_${SELECTED_EPOCH}.json"

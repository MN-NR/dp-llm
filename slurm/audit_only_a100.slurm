#!/usr/bin/env bash
#SBATCH --job-name=dpllm-audit-only
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Run ONLY the CanaryBench privacy audit for one or more already-selected
# checkpoints (training + U_val selection already done). Recovery job for
# runs whose in-job Phase 2 was killed at the walltime wall.
#
#   sbatch -M ascend --export=ALL,SPECS="gemma3_1b:baseline_seed42_epochs20:20:gemma3_baseline_nodp_seed42 gemma3_1b:baseline_seed43_epochs20:20:gemma3_baseline_nodp_seed43" \
#     slurm/audit_only_a100.slurm
#
# SPECS = space-separated  MODELKEY:CKPT_SUBDIR:EPOCH:TAG  tuples.
# ---------------------------------------------------------------------------
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
: "${SPECS:?set SPECS= via --export}"

DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
nvidia-smi

for spec in $SPECS; do
  IFS=: read -r MK SUB EP TAG <<< "$spec"
  CKPT_ROOT="$DPLLM_ROOT/checkpoints/instruction_recall/$MK/$SUB"
  echo
  echo "############ audit  $TAG  ($MK  $SUB  epoch $EP) ############"
  export DPLLM_MODEL_KEY="$MK"
  python slurm/privacy_audit.py \
      --model-key "$MK" \
      --ckpt-root "$CKPT_ROOT" \
      --epoch "$EP" \
      --member-path "$DATA_DIR/member_canaries.jsonl" \
      --reference-path "$DATA_DIR/attack_eval.jsonl" \
      --val-path "$DATA_DIR/combined_val.jsonl" \
      --device cuda \
      --out "$OUT_DIR/privacy_audit_${TAG}_epoch_${EP}.json"
done
echo
echo ">> all audits done."

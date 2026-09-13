#!/usr/bin/env bash
#SBATCH --job-name=dpllm-reaudit-a100
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
# Re-run privacy_audit.py for the three already-selected checkpoints, now
# that privacy_audit.py's utility re-confirmation uses the SAME strict
# evaluator as checkpoint selection (classification_eval_methods) on FULL
# val, instead of metrics.utility.task.evaluate_group_a (fuzzy matcher, a
# 500-record file-order group-A subset -> only ~264 classification). See
# the audit JSON's utility_full_val.evaluator field.
#
# Selected epochs (unchanged -- selection ranking was already internally
# consistent): eps3 -> epoch 20, eps5 -> epoch 16, eps8 -> epoch 18.
#
#   sbatch -M ascend slurm/reaudit_a100.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

DATA_DIR="$DPLLM_ROOT/instruction_recall_dataset/dataset/processed"
OUT_DIR="$DPLLM_ROOT/results/instruction_recall"
CKPT_BASE="$DPLLM_ROOT/checkpoints/instruction_recall/llama3_2_1b_it"

nvidia-smi -L

run_audit () {
    local eps="$1" epoch="$2" tag="$3"
    local ckpt_root="$CKPT_BASE/dp_eps${eps}_seed42_batch128_epochs20_2gpu"
    echo
    echo "=================================================================="
    echo "  RE-AUDIT  eps=${eps}  epoch_${epoch}"
    echo "=================================================================="
    python slurm/privacy_audit.py \
        --ckpt-root "$ckpt_root" \
        --epoch "$epoch" \
        --member-path "$DATA_DIR/member_canaries.jsonl" \
        --reference-path "$DATA_DIR/attack_eval.jsonl" \
        --val-path "$DATA_DIR/combined_val.jsonl" \
        --device cuda \
        --out "$OUT_DIR/privacy_audit_${tag}_epoch_${epoch}.json"
}

run_audit 3.0 20 eps3
run_audit 5.0 16 eps5
run_audit 8.0 18 eps8

echo
echo ">> Done. Re-audited JSONs:"
ls -la "$OUT_DIR"/privacy_audit_eps*_epoch_*.json

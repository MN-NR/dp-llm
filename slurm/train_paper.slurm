#!/usr/bin/env bash
#SBATCH --job-name=dpllm-llama-eps5
#SBATCH --account=pmiu0248
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# THE PAPER RUN.  Llama-3.2-1B-Instruct, DP-LoRA, epsilon = 5.0, 2x H100.
# Do NOT submit until:
#   - slurm/train_test.slurm passed (tokenizer + 50-step smoke)
#   - slurm/bench_physical_batch.slurm ran and PHYS_BATCH below is set to the
#     largest/fastest stable physical batch it found
#   - the user has said go (per "Stop and report before launching")
#
# Frozen corrected pipeline: processed_v2 dataset, EOS-supervised training,
# canonical prompt formatter, corrected eval, seed 42, max_grad_norm 1.0
# (from config), same accountant, LoRA config from config.py, global batch
# 128, LR 2e-5, <= 20 epochs, test set untouched. Every epoch checkpoint is
# saved; checkpoint SELECTION is a separate post-hoc step on validation
# utility only (see slurm/README.md).
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = 128 global logical batch
GRAD_ACCUM=8
PHYS_BATCH=64  
LR=2e-5
EPS=5.0
SEED=42
EPOCHS=20

if [[ "$PHYS_BATCH" == "__SET_FROM_BENCHMARK__" ]]; then
    echo "ERROR: set PHYS_BATCH from slurm/bench_physical_batch.slurm results first." >&2
    exit 2
fi

echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py
echo

nvidia-smi
echo ">> global batch $GLOBAL_BATCH  (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  eps $EPS  epochs $EPOCHS  seed $SEED"

srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
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

echo ">> training finished (rc=$?).  Checkpoints under checkpoints/instruction_recall/llama3_2_1b_it/"
echo ">> NEXT: checkpoint selection on validation utility only, then the privacy audit. See slurm/README.md."

#!/usr/bin/env bash
#SBATCH --job-name=dpllm-llama-eps3-a100
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Same as train_paper_a100.slurm but epsilon = 3.0 (tighter privacy budget --
# same batch/LR/epoch/seed/LoRA config, Opacus just calibrates a larger
# noise multiplier). Writes to a DIFFERENT checkpoint dir than the eps=5 run
# (dp_eps3.0_seed42_batch128_epochs20_2gpu/), so no collision with it -- the
# two can run concurrently.
#
#   sbatch -M ascend slurm/train_paper_a100_eps3.slurm
#
# Same NCCL_P2P_DISABLE=1 / pb=16 / --cpus-per-task=32-to-srun fixes as the
# eps=5 run -- see train_paper_a100.slurm's comments for why.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export NCCL_P2P_DISABLE=1

GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = 128 global logical batch
GRAD_ACCUM=8
PHYS_BATCH=16
LR=2e-5
EPS=3.0
SEED=42
EPOCHS=20

echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py
echo

nvidia-smi
echo ">> global batch $GLOBAL_BATCH  (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  eps $EPS  epochs $EPOCHS  seed $SEED"

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

echo ">> training finished (rc=$?).  Checkpoints under checkpoints/instruction_recall/llama3_2_1b_it/"
echo ">> NEXT: checkpoint selection on validation utility only, then the privacy audit. See slurm/README.md."

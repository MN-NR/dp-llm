#!/usr/bin/env bash
#SBATCH --job-name=dpllm-llama-eps5-a100
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
# THE PAPER RUN, A100 variant (Ascend cluster's `nextgen` partition instead
# of Cardinal's `gpu`). Same config, same output_dir/run_tag as
# train_paper.slurm (checkpoints/.../dp_eps5.0_seed42_batch128_epochs20_2gpu/)
# -- the Cardinal job (14170701) was cancelled, so this is the only run
# targeting that path. If you ever resubmit train_paper.slurm on Cardinal
# too, cancel one of them first to avoid both writing to the same dir.
#
# Submit with -M on the command line (cluster selection must happen before
# Slurm resolves this script's partition/QOS):
#   sbatch -M ascend slurm/train_paper_a100.slurm
#
# Do NOT submit until slurm/train_test_a100.slurm has passed (confirms A100
# VRAM + the pipeline on this cluster) and the user has said go.
#
# NCCL_P2P_DISABLE=1 required on Ascend: direct GPU P2P between these A100
# pairs hangs forever (negotiates fine, Init COMPLETE, transfer never
# completes -- confirmed via slurm/diag_a100_nccl.{py,slurm}). Disabling
# P2P routes through host memory (SHM) instead; barrier/all_reduce then
# complete in <1s. Transport choice only, doesn't touch DP-SGD math. NOT
# set in slurm/env.sh -- Cardinal's H100 P2P works fine as-is.
#
# PHYS_BATCH=16: train_test_a100.slurm confirmed these are A100-PCIE-40GB
# (not 80GB) and measured peak_gpu_mem_gb=20.9 at pb=16 (~19GB headroom).
# 50 real DPDDP steps took 84.6s = 1.69 s/opt-step (~2x H100's pb=32 rate --
# smaller physical batch, NCCL P2P disabled, A100 vs H100 raw throughput).
# At ~649 opt-steps/epoch x 20 epochs that's ~6.1h of pure training loop;
# with per-epoch val/canary/greedy-gen eval on top, budget ~8-10h -- hence
# --time=16:00:00 above (Ascend has abundant idle capacity, so a generous
# --time here doesn't hurt scheduling the way it did on Cardinal).
#
# Frozen corrected pipeline: seed 42, max_grad_norm 1.0 (from config), same
# accountant, LoRA config from config.py, global batch 128, LR 2e-5,
# <= 20 epochs, test set untouched. Every epoch checkpoint saved;
# checkpoint SELECTION is a separate post-hoc step on validation utility
# only (see slurm/README.md).
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
export NCCL_P2P_DISABLE=1

GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = 128 global logical batch
GRAD_ACCUM=8
PHYS_BATCH=16
LR=2e-5
EPS=5.0
SEED=42
EPOCHS=20

echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py
echo

nvidia-smi
echo ">> global batch $GLOBAL_BATCH  (bs $BATCH_SIZE x ga $GRAD_ACCUM)  phys $PHYS_BATCH  LR $LR  eps $EPS  epochs $EPOCHS  seed $SEED"

# --cpus-per-task passed explicitly -- see train_test_a100.slurm's comment
# (Ascend's Slurm derives a conflicting SLURM_CPUS_PER_TASK otherwise).
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

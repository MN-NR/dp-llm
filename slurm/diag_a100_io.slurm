#!/usr/bin/env bash
#SBATCH --job-name=dpllm-diag-io
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Diagnostic only. train_test_a100.slurm hung 10+ min inside from_pretrained
# on two different Ascend nodes (a0200, a0176) before the NCCL watchdog
# killed it. This isolates: raw GPFS scratch read throughput, and a
# single-process (no DDP/NCCL) from_pretrained + .to(cuda) timing, so a
# stall shows up directly instead of downstream as an opaque NCCL timeout.
#
#   sbatch -M ascend slurm/diag_a100_io.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

echo "### node ###"
hostname
nvidia-smi -L

echo
echo "### diagnostic ###"
python slurm/diag_a100_io.py

#!/usr/bin/env bash
#SBATCH --job-name=dpllm-diag-concurrent
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Diagnostic only. Isolates whether 2 concurrent from_pretrained() calls on
# the same Ascend node hang, independent of torchrun/NCCL (plain subprocess,
# 3-min hard cap so this doesn't need the full 10-min NCCL watchdog to
# answer). See diag_a100_io.slurm (ruled out: raw FS throughput, single-
# process load speed -- both fine).
#
#   sbatch -M ascend slurm/diag_a100_concurrent.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

hostname
nvidia-smi -L
python slurm/diag_a100_concurrent.py

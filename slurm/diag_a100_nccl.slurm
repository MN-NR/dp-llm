#!/usr/bin/env bash
#SBATCH --job-name=dpllm-diag-nccl
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Diagnostic only. Bare NCCL init + barrier() between 2 GPUs on one Ascend
# node -- the exact first collective train_dp_ddp.py does. NCCL_DEBUG=INFO
# for visibility; 60s init timeout so a hang fails fast instead of the
# usual 600s. See diag_a100_io.slurm / diag_a100_concurrent.slurm (both
# ruled out: FS throughput, single- and concurrent-process model loading).
#
#   sbatch -M ascend slurm/diag_a100_nccl.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

hostname
nvidia-smi -L
export NCCL_DEBUG=INFO
# NCCL_CUMEM_ENABLE=0 (previous test) still hung, just via P2P/IPC instead
# of P2P/CUMEM -- so it's not the CUMEM allocator, it's P2P itself that
# doesn't actually work on this node pair (topology says it should; the
# real transfer silently hangs -- classic IOMMU/ACS-blocked peer access
# symptom). Disable P2P entirely so NCCL stages through host memory.
export NCCL_P2P_DISABLE=1

srun --cpus-per-task=32 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    slurm/diag_a100_nccl.py

#!/usr/bin/env bash
#SBATCH --job-name=dpllm-traintest-a100
#SBATCH --account=pmiu0248
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# A100 SMOKE TEST -- not the paper run. Same as train_test.slurm but targets
# the Ascend cluster's A100s instead of Cardinal's H100s (idle A100 capacity
# was far higher than idle H100 capacity at the time this was written -- see
# EXPERIMENT_LOG.md). Must submit with `-M ascend` on the command line (NOT
# just an #SBATCH --clusters= line -- cluster selection has to happen before
# Slurm resolves this script's partition/QOS):
#
#   sbatch -M ascend slurm/train_test_a100.slurm
#
# Uses `nextgen` (not `debug-nextgen` -- its `ascend-debug` QOS caps 1
# GPU/node, same QOSMaxGRESPerNode trap as Cardinal's `debug`). `nextgen`
# allows <=3 GPU/job and had ~95 idle A100 GPUs against ~13 GPU-requesting
# pending jobs when checked -- should still start fast despite the 7-day
# partition limit (we only request 45 min via --time).
#
# Primary purpose beyond the usual smoke checks: CONFIRM ACTUAL A100 VRAM.
# Slurm's gres doesn't expose per-GPU memory size; nvidia-smi here is the
# first real answer. If it's 40GB (not 80GB), physical batch 32 (not 64)
# is what fits -- see the peak_gpu_mem_gb this run reports and the pb 32
# vs 64 numbers from the H100 benchmark (38.0GB vs 72.3GB peak) as a guide.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

# Ascend-specific: direct GPU P2P between these A100 pairs hangs forever
# (NCCL negotiates it as available, Init COMPLETE, then the actual transfer
# never completes -- confirmed via slurm/diag_a100_nccl.{py,slurm}: same
# hang via P2P/CUMEM and via legacy P2P/IPC, fixed by disabling P2P so NCCL
# routes through host memory (SHM) instead -- barrier/all_reduce then
# complete in <1s). Purely a transport choice, doesn't touch DP-SGD math.
# NOT set in slurm/env.sh: Cardinal's H100 P2P works fine and disabling it
# there would just be a needless slowdown.
export NCCL_P2P_DISABLE=1

echo "### nvidia-smi (confirms A100 VRAM size) ###"
nvidia-smi
echo "### torch sees CUDA ###"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print([torch.cuda.get_device_properties(i).total_memory/1e9 for i in range(torch.cuda.device_count())], 'GB per device')"

echo
echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py

echo
echo "### STEP 1: verify Llama tokenizer + EOS formatting ###"
python slurm/verify_llama_tokenizer.py --hf-token "$HF_TOKEN"

echo
echo "### STEP 2: 50-step DPDDP smoke run ###"
echo "### confirmed A100-PCIE-40GB -> pb=32 (38GB on H100) is too tight; using pb=16 ###"
GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = global logical batch
GRAD_ACCUM=8
PHYS_BATCH=16
LR=2e-5
EPS=5.0
SEED=42

# --cpus-per-task passed explicitly: Ascend's Slurm (25.05.4) derives a
# conflicting SLURM_CPUS_PER_TASK vs SLURM_TRES_PER_TASK from #SBATCH alone
# on these nodes ("srun: fatal: cpus-per-task set by two different
# environment variables") -- giving srun the value directly sidesteps it.
srun --cpus-per-task=32 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    finetune/train_dp_ddp.py \
        --model llama3_2_1b_it \
        --epsilon "$EPS" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --physical-batch-size "$PHYS_BATCH" \
        --learning-rate "$LR" \
        --max-steps 50 \
        --hf_token "$HF_TOKEN"

echo
echo "### smoke test finished (rc=$?) ###"
echo "### Check the 'peak_gpu_mem_gb' in results/instruction_recall/*.json and this job's peak_gpu_mem_gb log line to decide pb=32 vs pb=64 for train_paper_a100.slurm ###"

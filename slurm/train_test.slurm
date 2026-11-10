#!/usr/bin/env bash
#SBATCH --job-name=dpllm-traintest
#SBATCH --account=pmiu0248
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# SMOKE TEST -- not the paper run.
#
#   1. verify Llama-3.2-1B-Instruct tokenizer + EOS-supervised formatting
#   2. run finetune/train_dp_ddp.py for --max-steps 50 on 2x H100 via
#      torchrun, at the paper config (global batch 128, LR 2e-5, eps 5.0,
#      seed 42), physical batch 32.
#
# Confirms the DPDDP path, Opacus accountant, dataloader, checkpoint dir and
# NCCL all work end to end before committing to the 20-epoch job. Nothing
# here is a valid training result.
#
# Submit:  sbatch slurm/train_test.slurm
# Must be `gpu` partition, not `debug`: the cardinal-debug QOS caps jobs at
# 1 GPU/node (QOSMaxGRESPerNode) and this needs 2. `gpu` allows <=4 GPU/job.
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

echo "### nvidia-smi ###"
nvidia-smi
echo "### torch sees CUDA ###"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

echo
echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py

echo
echo "### STEP 1: verify Llama tokenizer + EOS formatting ###"
python slurm/verify_llama_tokenizer.py --hf-token "$HF_TOKEN"

echo
echo "### STEP 2: 50-step DPDDP smoke run ###"
GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = global logical batch
GRAD_ACCUM=8
PHYS_BATCH=32
LR=2e-5
EPS=5.0
SEED=42

srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
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

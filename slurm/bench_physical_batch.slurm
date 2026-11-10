#!/usr/bin/env bash
#SBATCH --job-name=dpllm-benchpb
#SBATCH --account=pmiu0248
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out
# ---------------------------------------------------------------------------
# Step 2 of the checklist: benchmark PHYSICAL batch 32 vs 64 at a fixed
# GLOBAL logical batch of 128, on 2x H100, to pick the largest/fastest
# stable physical batch for the 20-epoch run.
#
# For each physical batch size it runs train_dp_ddp.py in its built-in
# throughput mode (--max-steps 30: does real DP-SGD steps, skips
# validation/canary-eval, emits BENCH_LOOP_START/END and peak_gpu_mem_gb),
# while sampling nvidia-smi for utilisation and VRAM. Reports:
#   sec / optimizer-step, mean+peak GPU util %, peak VRAM (GB), non-finite steps
#
# Nothing here is a training result; checkpoints from --max-steps runs are
# throwaway. Submit:  sbatch slurm/bench_physical_batch.slurm
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

echo "### PREFLIGHT: HF gated-repo content access ###"
python slurm/check_hf_gated.py
echo

RESULTS_DIR="$DPLLM_ROOT/slurm/logs/bench_${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULTS_DIR"

GLOBAL_BATCH=128
BATCH_SIZE=16          # batch_size * grad_accum = 128 global logical
GRAD_ACCUM=8
LR=2e-5
EPS=5.0
SEED=42
STEPS=30

nvidia-smi
echo

for PHYS in 32 64; do
    echo "==================================================================="
    echo "  PHYSICAL BATCH = $PHYS   (global logical batch = $GLOBAL_BATCH)"
    echo "==================================================================="
    RUN_LOG="$RESULTS_DIR/pb${PHYS}.log"
    SMI_LOG="$RESULTS_DIR/pb${PHYS}.nvsmi.csv"

    # background GPU sampler @ 1 Hz
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
               --format=csv,noheader,nounits -l 1 > "$SMI_LOG" &
    SMI_PID=$!

    set +e
    srun torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        finetune/train_dp_ddp.py \
            --model llama3_2_1b_it \
            --epsilon "$EPS" --seed "$SEED" \
            --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
            --physical-batch-size "$PHYS" \
            --learning-rate "$LR" \
            --max-steps "$STEPS" \
            --hf_token "$HF_TOKEN" 2>&1 | tee "$RUN_LOG"
    RC=$?
    set -e

    kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true
    echo "  (run rc=$RC)"
    echo
done

echo "==================================================================="
echo "  SUMMARY"
echo "==================================================================="
python - "$RESULTS_DIR" <<'PY'
import csv, glob, os, re, sys
d = sys.argv[1]
for log in sorted(glob.glob(os.path.join(d, "pb*.log"))):
    pb = re.search(r"pb(\d+)\.log", log).group(1)
    txt = open(log, errors="replace").read()
    m0 = re.search(r"BENCH_LOOP_START ([\d.]+)", txt)
    m1 = re.search(r"BENCH_LOOP_END ([\d.]+) steps=(\d+) nonfinite=(\d+)", txt)
    peak_mem = re.search(r'"peak_gpu_mem_gb":\s*([\d.]+)', txt) or \
               re.search(r"peak_gpu_mem_gb['\"]?\s*[:=]\s*([\d.]+)", txt)
    noise = re.search(r'"noise_multiplier":\s*([\d.]+)', txt)
    oom = "CUDA out of memory" in txt or "OutOfMemoryError" in txt
    line = f"  pb={pb:>3}  "
    if m0 and m1:
        dt = float(m1.group(1)) - float(m0.group(1))
        steps = int(m1.group(2))
        line += f"sec/opt-step={dt/max(steps,1):7.3f}  steps={steps}  nonfinite={m1.group(3)}  "
    else:
        line += "sec/opt-step=   n/a  (no BENCH markers)  "
    line += f"peak_alloc_GB={peak_mem.group(1) if peak_mem else '?':>6}  "
    line += f"noise={noise.group(1) if noise else '?'}  "
    if oom:
        line += "OOM!"
    # nvidia-smi derived util/mem
    smi = log.replace(".log", ".nvsmi.csv")
    if os.path.exists(smi):
        utils, mems = [], []
        for row in csv.reader(open(smi)):
            if len(row) >= 4:
                try:
                    utils.append(float(row[2])); mems.append(float(row[3]))
                except ValueError:
                    pass
        if utils:
            line += f"\n           GPU util: mean={sum(utils)/len(utils):5.1f}%  peak={max(utils):5.1f}%   "
            line += f"VRAM used: peak={max(mems)/1024:5.1f} GB"
    print(line)
print()
print("  Pick the largest physical batch that: (a) did not OOM, (b) had 0 non-finite")
print("  steps, (c) gives the lowest sec/opt-step. Set it as PHYS_BATCH in train_paper.slurm.")
PY

#!/usr/bin/env bash
set -euo pipefail

cd /home/noora/dp-llm/dp-llm
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON=/home/noora/miniconda3/envs/dp_gpu/bin/python
OUT=/home/noora/dp-llm/dp-llm/results/eos_fixed/non_dp/gemma_2b_it
MANIFEST="$OUT/search_manifest.json"
CANDIDATE=lr_2e-05_ep_08_gb_64

"$PYTHON" -m canarybench.tuning run-candidate \
  --manifest "$MANIFEST" \
  --candidate-id "$CANDIDATE" \
  2>&1 | tee "$OUT/tmux_training.log"

"$PYTHON" rebuttal_experiments/postprocess_gemma2_nondp.py \
  2>&1 | tee "$OUT/tmux_postprocess.log"

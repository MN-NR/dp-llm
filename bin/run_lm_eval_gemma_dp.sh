#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run lm-eval-harness utility eval for Gemma checkpoints:
  - gemma_2b_it: baseline + DP eps 3/5/8
  - gemma3_1b:  baseline + DP eps 3/5/8

Usage:
  bin/run_lm_eval_gemma_dp.sh [options]

Options:
  --models <list>          Default: gemma_2b_it,gemma3_1b
                           Allowed: gemma_2b_it, gemma3_1b, all
  --eps <list>             Default: 3,5,8   (DP only; baseline always runs)
  --which <final|best>     Default: final   (DP checkpoints)
  --tasks <comma_list>     Default: hellaswag,arc_easy,boolq,piqa
  --device <device>        Default: cuda
  --batch_size <n>         Default: 4
  --dtype <dtype>          Default: bfloat16
  --out_dir <dir>          Default: results/lm_eval_utility

  --base_gemma2 <id>       Default: google/gemma-2b-it
  --base_gemma3 <id>       Default: google/gemma-3-1b-it

Example:
  bin/run_lm_eval_gemma_dp.sh --which final --device cuda --batch_size 4

Tip:
  For a fully generalized single-checkpoint runner, use:
    bin/run_lm_eval_ckpt.sh --base <hf_id> --model_key <dir> --variant <baseline|dp> [--eps N]
EOF
}

models="gemma_2b_it,gemma3_1b"
eps_list="3,5,8"
which="final"
tasks="hellaswag,arc_easy,boolq,piqa"
device="cuda"
batch_size="4"
dtype="bfloat16"
out_dir="results/lm_eval_utility"
base_gemma2="google/gemma-2-2b-it"
base_gemma3="google/gemma-3-1b-it"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) models="${2:-}"; shift 2 ;;
    --eps) eps_list="${2:-}"; shift 2 ;;
    --which) which="${2:-}"; shift 2 ;;
    --tasks) tasks="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --batch_size) batch_size="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --out_dir) out_dir="${2:-}"; shift 2 ;;
    --base_gemma2) base_gemma2="${2:-}"; shift 2 ;;
    --base_gemma3) base_gemma3="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$which" != "final" && "$which" != "best" ]]; then
  echo "error: --which must be 'final' or 'best' (got: $which)" >&2
  exit 2
fi

mkdir -p "$out_dir"

resolve_base() {
  case "$1" in
    gemma_2b_it) echo "$base_gemma2" ;;
    gemma3_1b) echo "$base_gemma3" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local model_key="$1"
  local base_id="$2"
  local ckpt_dir="$3"
  local tag="$4"

  if [[ ! -d "$ckpt_dir" ]]; then
    echo "error: checkpoint dir not found: $ckpt_dir" >&2
    exit 2
  fi

  local ts out_file
  ts="$(date +%Y%m%d_%H%M%S)"
  out_file="${out_dir}/${model_key}_${tag}_${ts}.log"
  echo "[lm-eval] model=${model_key} tag=${tag} ckpt=${ckpt_dir}"
  echo "[lm-eval] writing: ${out_file}"

  bin/run_lm_eval_utility.sh \
    --base "$base_id" \
    --peft "$ckpt_dir" \
    --tasks "$tasks" \
    --device "$device" \
    --batch_size "$batch_size" \
    --dtype "$dtype" \
    |& tee "$out_file"
}

models_expanded="$models"
if [[ "$models" == "all" ]]; then
  models_expanded="gemma_2b_it,gemma3_1b"
fi

IFS=',' read -r -a model_values <<<"$models_expanded"
for model_key in "${model_values[@]}"; do
  model_key="${model_key// /}"
  [[ -z "$model_key" ]] && continue

  base_id="$(resolve_base "$model_key")" || {
    echo "error: unknown model in --models: $model_key" >&2
    exit 2
  }

  run_one "$model_key" "$base_id" "checkpoints/${model_key}/baseline/final" "baseline_final"

  IFS=',' read -r -a eps_values <<<"$eps_list"
  for eps in "${eps_values[@]}"; do
    eps="${eps// /}"
    [[ -z "$eps" ]] && continue
    ckpt_dir="checkpoints/${model_key}/dp_eps${eps}.0/${which}"
    if [[ ! -d "$ckpt_dir" ]]; then
      ckpt_dir="checkpoints/${model_key}/dp_eps${eps}/${which}"
    fi
    if [[ ! -d "$ckpt_dir" ]]; then
      echo "error: checkpoint not found for model=${model_key} eps=${eps} which=${which}: tried:" >&2
      echo "  checkpoints/${model_key}/dp_eps${eps}.0/${which}" >&2
      echo "  checkpoints/${model_key}/dp_eps${eps}/${which}" >&2
      exit 2
    fi

    run_one "$model_key" "$base_id" "$ckpt_dir" "dp_eps${eps}_${which}"
  done
done

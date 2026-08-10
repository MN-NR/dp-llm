#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  bin/run_lm_eval_utility.sh --base <hf_model_id_or_path> --peft <checkpoint_dir> [options]

Options:
  --tasks <comma_list>        Default: hellaswag,arc_easy,boolq,piqa
  --device <device>          Default: cuda
  --batch_size <n>           Default: 4
  --dtype <dtype>            Default: bfloat16
  --extra_model_args <args>  Extra hf model_args, appended (comma-separated)

Example:
  bin/run_lm_eval_utility.sh --base google/gemma-2-2b-it --peft checkpoints/gemma_2b_it/baseline/final
EOF
}

base=""
peft=""
tasks="hellaswag,arc_easy,boolq,piqa"
device="cuda"
batch_size="4"
dtype="bfloat16"
extra_model_args=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) base="${2:-}"; shift 2 ;;
    --peft) peft="${2:-}"; shift 2 ;;
    --tasks) tasks="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --batch_size) batch_size="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --extra_model_args) extra_model_args="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$base" || -z "$peft" ]]; then
  echo "error: --base and --peft are required" >&2
  usage
  exit 2
fi

if [[ ! -d "$peft" ]]; then
  echo "error: peft checkpoint dir not found: $peft" >&2
  exit 2
fi

model_args="pretrained=${base},peft=${peft},dtype=${dtype}"
if [[ -n "$extra_model_args" ]]; then
  model_args="${model_args},${extra_model_args}"
fi

exec "$(dirname "$0")/bashlm_eval" \
  --model hf \
  --model_args "${model_args}" \
  --tasks "${tasks}" \
  --device "${device}" \
  --batch_size "${batch_size}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
General lm-eval-harness utility runner for this repo's checkpoint layout.

It resolves a PEFT checkpoint dir like:
  checkpoints/<model_key>/baseline/final
  checkpoints/<model_key>/dp_eps<eps>.0/<final|best>

Usage:
  bin/run_lm_eval_ckpt.sh --base <hf_id_or_path> --model_key <key> --variant <baseline|dp> [options]

Required:
  --base <hf_id_or_path>     HF base model id/path (e.g., google/gemma-2b-it)
  --model_key <key>          Checkpoint folder under checkpoints/ (e.g., gemma_2b_it)
  --variant <baseline|dp>    Which checkpoint type to run

Options:
  --which <final|best>       Default: final
  --eps <n>                  Required when --variant dp (e.g., 3, 5, 8)
  --ckpt_root <dir>          Default: checkpoints
  --tasks <comma_list>       Default: hellaswag,arc_easy,boolq,piqa
  --device <device>          Default: cuda
  --batch_size <n>           Default: 4
  --dtype <dtype>            Default: bfloat16
  --out_dir <dir>            Default: results/lm_eval_utility

Examples:
  bin/run_lm_eval_ckpt.sh --base google/gemma-2b-it --model_key gemma_2b_it --variant baseline
  bin/run_lm_eval_ckpt.sh --base google/gemma-2b-it --model_key gemma_2b_it --variant dp --eps 5 --which best
EOF
}

base=""
model_key=""
variant=""
which="final"
eps=""
ckpt_root="checkpoints"
tasks="hellaswag,arc_easy,boolq,piqa"
device="cuda"
batch_size="4"
dtype="bfloat16"
out_dir="results/lm_eval_utility"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) base="${2:-}"; shift 2 ;;
    --model_key) model_key="${2:-}"; shift 2 ;;
    --variant) variant="${2:-}"; shift 2 ;;
    --which) which="${2:-}"; shift 2 ;;
    --eps) eps="${2:-}"; shift 2 ;;
    --ckpt_root) ckpt_root="${2:-}"; shift 2 ;;
    --tasks) tasks="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --batch_size) batch_size="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --out_dir) out_dir="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$base" || -z "$model_key" || -z "$variant" ]]; then
  echo "error: --base, --model_key, and --variant are required" >&2
  usage
  exit 2
fi

if [[ "$which" != "final" && "$which" != "best" ]]; then
  echo "error: --which must be 'final' or 'best' (got: $which)" >&2
  exit 2
fi

case "$variant" in
  baseline|dp) ;;
  *) echo "error: --variant must be 'baseline' or 'dp' (got: $variant)" >&2; exit 2 ;;
esac

mkdir -p "$out_dir"

ckpt_dir=""
tag=""
if [[ "$variant" == "baseline" ]]; then
  ckpt_dir="${ckpt_root}/${model_key}/baseline/final"
  tag="baseline_final"
else
  if [[ -z "$eps" ]]; then
    echo "error: --eps is required when --variant dp" >&2
    exit 2
  fi
  ckpt_dir="${ckpt_root}/${model_key}/dp_eps${eps}.0/${which}"
  if [[ ! -d "$ckpt_dir" ]]; then
    ckpt_dir="${ckpt_root}/${model_key}/dp_eps${eps}/${which}"
  fi
  tag="dp_eps${eps}_${which}"
fi

if [[ ! -d "$ckpt_dir" ]]; then
  echo "error: checkpoint dir not found: $ckpt_dir" >&2
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"
out_file="${out_dir}/${model_key}_${tag}_${ts}.log"
echo "[lm-eval] model_key=${model_key} tag=${tag} ckpt=${ckpt_dir}"
echo "[lm-eval] base=${base}"
echo "[lm-eval] writing: ${out_file}"

bin/run_lm_eval_utility.sh \
  --base "$base" \
  --peft "$ckpt_dir" \
  --tasks "$tasks" \
  --device "$device" \
  --batch_size "$batch_size" \
  --dtype "$dtype" \
  |& tee "$out_file"


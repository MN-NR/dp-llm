#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run lm-eval-harness utility eval for VaultGemma DP checkpoints (eps 3/5/8).

Usage:
  bin/run_lm_eval_vault_gemma_dp.sh [options]

Options:
  --base <hf_id_or_path>   Default: google/vaultgemma-1b
  --which <final|best>     Default: final
  --eps <list>             Default: 3,5,8
  --tasks <comma_list>     Default: hellaswag,arc_easy,boolq,piqa
  --device <device>        Default: cuda
  --batch_size <n>         Default: 4
  --dtype <dtype>          Default: bfloat16
  --out_dir <dir>          Default: results/lm_eval_utility

Example:
  bin/run_lm_eval_vault_gemma_dp.sh --which final --device cuda --batch_size 4
EOF
}

base="google/vaultgemma-1b"
which="final"
eps_list="3,5,8"
tasks="hellaswag,arc_easy,boolq,piqa"
device="cuda"
batch_size="4"
dtype="bfloat16"
out_dir="results/lm_eval_utility"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) base="${2:-}"; shift 2 ;;
    --which) which="${2:-}"; shift 2 ;;
    --eps) eps_list="${2:-}"; shift 2 ;;
    --tasks) tasks="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --batch_size) batch_size="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --out_dir) out_dir="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$which" != "final" && "$which" != "best" ]]; then
  echo "error: --which must be 'final' or 'best' (got: $which)" >&2
  exit 2
fi

mkdir -p "$out_dir"

IFS=',' read -r -a eps_values <<<"$eps_list"
for eps in "${eps_values[@]}"; do
  eps="${eps// /}"
  [[ -z "$eps" ]] && continue
  ckpt_dir="checkpoints/vault_gemma/dp_eps${eps}.0/${which}"
  if [[ ! -d "$ckpt_dir" ]]; then
    ckpt_dir="checkpoints/vault_gemma/dp_eps${eps}/${which}"
  fi
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "error: checkpoint not found for eps=${eps} (${which}): tried:" >&2
    echo "  checkpoints/vault_gemma/dp_eps${eps}.0/${which}" >&2
    echo "  checkpoints/vault_gemma/dp_eps${eps}/${which}" >&2
    exit 2
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  out_file="${out_dir}/vault_gemma_dp_eps${eps}_${which}_${ts}.log"
  echo "[lm-eval] eps=${eps} which=${which} ckpt=${ckpt_dir}"
  echo "[lm-eval] writing: ${out_file}"

  bin/run_lm_eval_utility.sh \
    --base "$base" \
    --peft "$ckpt_dir" \
    --tasks "$tasks" \
    --device "$device" \
    --batch_size "$batch_size" \
    --dtype "$dtype" \
    |& tee "$out_file"
done


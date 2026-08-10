#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-baseline}"
MODEL="${MODEL:-gemma_2b_it}"
EPSILON="${EPSILON:-8.0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/enron_data/processed}"

TRAIN_PATH="$DATA_DIR/combined_train.jsonl"
VAL_PATH="$DATA_DIR/combined_val.jsonl"
CANARY_PATH="$DATA_DIR/member_canaries.jsonl"
REFERENCE_PATH="$DATA_DIR/attack_eval.jsonl"
CANARY_REGISTRY_PATH="$DATA_DIR/canary_registry.json"

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing training file: $TRAIN_PATH" >&2
  exit 1
fi

if [[ ! -f "$VAL_PATH" ]]; then
  echo "Missing validation file: $VAL_PATH" >&2
  exit 1
fi

case "$MODE" in
  baseline)
    exec python "$ROOT_DIR/finetune/train_baseline.py" \
      --model "$MODEL" \
      --train-path "$TRAIN_PATH" \
      --val-path "$VAL_PATH" \
      --canary-path "$CANARY_PATH" \
      --reference-path "$REFERENCE_PATH" \
      --canary-registry-path "$CANARY_REGISTRY_PATH"
    ;;
  pilot)
    exec python "$ROOT_DIR/finetune/train_baseline.py" \
      --model "$MODEL" \
      --pilot \
      --train-path "$TRAIN_PATH" \
      --val-path "$VAL_PATH" \
      --canary-path "$CANARY_PATH" \
      --reference-path "$REFERENCE_PATH" \
      --canary-registry-path "$CANARY_REGISTRY_PATH"
    ;;
  dp)
    exec python "$ROOT_DIR/finetune/train_dp.py" \
      --model "$MODEL" \
      --epsilon "$EPSILON" \
      --train-path "$TRAIN_PATH" \
      --val-path "$VAL_PATH" \
      --canary-path "$CANARY_PATH" \
      --reference-path "$REFERENCE_PATH" \
      --canary-registry-path "$CANARY_REGISTRY_PATH"
    ;;
  dp-pilot)
    exec python "$ROOT_DIR/finetune/train_dp.py" \
      --model "$MODEL" \
      --epsilon "$EPSILON" \
      --pilot \
      --train-path "$TRAIN_PATH" \
      --val-path "$VAL_PATH" \
      --canary-path "$CANARY_PATH" \
      --reference-path "$REFERENCE_PATH" \
      --canary-registry-path "$CANARY_REGISTRY_PATH"
    ;;
  *)
    echo "Usage: $0 [baseline|pilot|dp|dp-pilot]" >&2
    echo "Optional env vars: MODEL=gemma_2b_it EPSILON=8.0 DATA_DIR=/path/to/processed" >&2
    exit 1
    ;;
esac

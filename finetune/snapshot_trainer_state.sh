#!/bin/bash

set -euo pipefail

SOURCE_DIR="${1:-/home/noora/dp-llm/checkpoints/vault_gemma/baseline}"
DEST_DIR="${2:-/home/noora/dp-llm/results/live_snapshots}"
INTERVAL_SECONDS="${3:-300}"

mkdir -p "$DEST_DIR"

while true; do
    latest_checkpoint="$(find "$SOURCE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"

    if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint/trainer_state.json" ]; then
        checkpoint_name="$(basename "$latest_checkpoint")"
        cp "$latest_checkpoint/trainer_state.json" "$DEST_DIR/${checkpoint_name}_latest_trainer_state.json"
        cp "$latest_checkpoint/trainer_state.json" "$DEST_DIR/${checkpoint_name}_trainer_state_$(date -u +%Y%m%d_%H%M%S).json"
    fi

    sleep "$INTERVAL_SECONDS"
done

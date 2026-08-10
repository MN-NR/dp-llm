#!/usr/bin/env python3
"""Evaluate the four unadapted base models with test_model.py's exact evaluator."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.config import MODELS  # noqa: E402
from finetune.test_model import load_model_and_tokenizer, run_evaluation  # noqa: E402

MODEL_KEYS = ("gemma_2b_it", "vault_gemma", "gemma3_1b", "llama3_2_1b_it")
DEFAULT_DATA_DIR = REPO_ROOT / "instruction_recall_dataset/dataset/processed"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/base_model_evaluations.json"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-batch-size", type=int, default=8)
    parser.add_argument("--metrics-max-samples", type=int, default=0, help="0 evaluates every record")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    payload = {
        "generated_at": timestamp(),
        "data_dir": str(args.data_dir),
        "generation_config": {
            "source": "finetune/test_model.py",
            "do_sample": False,
            "num_beams": 1,
            "task_max_tokens": "TASK_MAX_TOKENS",
        },
        "models": {},
    }

    for model_key in tqdm(args.models, desc="Evaluating base models", unit="model"):
        print(f"[{timestamp()}] Loading base model {model_key}")
        model, tokenizer, cfg = load_model_and_tokenizer(model_key, checkpoint_path=None)
        evaluations = []
        for dataset_type in tqdm(("test", "val"), desc=model_key, unit="split", leave=False):
            evaluations.append(
                run_evaluation(
                    model_key=model_key,
                    checkpoint_path=None,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    dataset_type=dataset_type,
                    device=args.device,
                    metrics_max_samples=args.metrics_max_samples,
                    metrics_batch_size=args.metrics_batch_size,
                    num_preview_samples=5,
                    save_all_samples=False,
                    sample_seed=args.sample_seed,
                    data_dir=str(args.data_dir),
                )
            )
        payload["models"][model_key] = {
            "model_name": MODELS[model_key].model_name,
            "checkpoint": None,
            "device": args.device,
            "completed_at": timestamp(),
            "evaluations": evaluations,
        }
        save_payload(args.output, payload)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["completed_at"] = timestamp()
    save_payload(args.output, payload)
    print(f"[{timestamp()}] Wrote {args.output}")


if __name__ == "__main__":
    main()

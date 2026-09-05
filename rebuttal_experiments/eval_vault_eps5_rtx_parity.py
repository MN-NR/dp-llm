"""Validation-only utility evaluation for the RTX VaultGemma epsilon-5 parity run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import rebuttal_experiments.postprocess_gemma2_nondp as established

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "instruction_recall_dataset/dataset/processed_v2"
BASE = Path(
    "/home/noora/.cache/huggingface/hub/models--google--vaultgemma-1b/"
    "snapshots/f9624dafc1760cb2f6039e86e12055d6559d7abb"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    os.environ.update(
        CUDA_VISIBLE_DEVICES="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    established.DATA = DATA
    established.BASE = BASE
    tasks = established.load_validation()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    details = []
    for epoch in args.epochs:
        checkpoint = args.checkpoint_root / "epoch_checkpoints" / f"epoch_{epoch:02d}"
        required = checkpoint / "adapter_model.safetensors"
        if not required.exists():
            raise FileNotFoundError(f"missing completed epoch checkpoint: {required}")
        print(f"[validation-only] epoch {epoch}: {checkpoint}", flush=True)
        _, detail = established.evaluate_epoch(checkpoint, epoch, tasks)
        detail["privacy_mode"] = "dp"
        detail["target_epsilon"] = 5.0
        detail["accounting_horizon_epochs"] = 20
        detail["test_set_touched"] = False
        with open(args.output_dir / f"epoch_{epoch:02d}.json", "w", encoding="utf-8") as handle:
            json.dump(detail, handle, indent=2)
        details.append(detail)

    payload = {
        "model": "vault_gemma",
        "purpose": "RTX epsilon-5 hardware/accounting/utility parity check",
        "selection_split": "validation",
        "test_set_touched": False,
        "epochs": details,
        "reference_result_supplied_by_user": {
            "epsilon_actual": 4.8387,
            "ag_news_accuracy": 0.825,
            "mnli_accuracy": 0.4167,
            "pii_unique_f1": 0.861,
            "memorization_gap": 0.007,
            "wbc_auc": 0.497,
            "note": "Final-result context only; memorization is not used for this parity decision."
        }
    }
    with open(args.output_dir / "validation_epochs_01_03.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

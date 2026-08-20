import json
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from canarybench.prompt_format import build_supervised_text, legacy_hardcoded_supervised_text


def evaluate_canaries_full_sequence(model, tokenizer, canary_path, device, max_samples=200, prompt_mode="canonical"):
    """
    Compute per-canary loss on the full prompt+target sequence.
    Only padding is masked; prompt tokens are included in the loss.

    prompt_mode:
      "canonical" (default) -- uses canarybench.prompt_format.build_supervised_text,
          the same chat-template-aware formatter used for training (finetune/dataset.py)
          and for the canonical canary-exposure eval path (metrics/base.py). This is a
          bugfix: previously this function always used a hardcoded "### Response:\n"
          template regardless of whether the tokenizer had a chat template, so for
          chat-template models (Gemma-it, VaultGemma, LLaMA-Instruct) the per-epoch
          canary loss logged here did not match the format the model was actually
          trained on. Note this bug only affected this logging/monitoring pathway
          (and the analogous helpers in benchmarks/finprivcanary) -- the actual
          training loss (finetune/dataset.py) and "best" checkpoint selection
          (val loss, also via finetune/dataset.py) were never affected by it.
      "legacy_hardcoded" -- byte-for-byte the old unconditional "### Response:\n"
          template, for reproducing historical logs/results exactly.
    """
    if not os.path.exists(canary_path):
        return [], 0.0
    if prompt_mode not in ("canonical", "legacy_hardcoded"):
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")

    was_training = model.training
    model.eval()
    results = []

    with open(canary_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = records[:max_samples]

    with torch.no_grad():
        for rec in records:
            if prompt_mode == "legacy_hardcoded":
                _, text = legacy_hardcoded_supervised_text(rec)
            else:
                _, text = build_supervised_text(tokenizer, rec)
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**enc, labels=labels)
            results.append({
                "id": rec.get("id", ""),
                "loss": round(outputs.loss.item(), 4),
                "canary": rec.get("canary", False),
                "source": rec.get("source", ""),
            })

    if was_training:
        model.train()

    avg_loss = sum(r["loss"] for r in results) / len(results) if results else 0.0
    return results, avg_loss

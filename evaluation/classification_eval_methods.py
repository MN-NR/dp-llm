"""AG News / MNLI evaluation repair: three scoring methods compared side by
side, so the existing raw-loss evaluator's label-prior bias can be diagnosed
and replaced with a validated alternative before any paper number is drawn
from it.

Evaluation only. Does not touch training, `finetune/dataset.py`, or
`canarybench/prompt_format.py`. Reuses `evaluation/loss_based_eval.py`'s
existing `_build_prompt` / `_avg_label_loss` / `_LABELS_BY_TASK` for the raw
and calibrated methods, so "raw-loss" here is bit-for-bit the same
computation the pipeline already uses -- not a reimplementation.

Three methods, per example:
  - raw:         argmin_label  L(label | prompt)                    (existing)
  - calibrated:  argmin_label  L(label | prompt) - L(label | null_prompt)
  - generation:  greedy-decode the model's own completion, parse the
                 first line, exact-match (after case/whitespace
                 normalization only -- no fuzzy matching) against the
                 task's label set; anything else is "invalid".

Null prompts (Method B): built from the *exact same literal instruction
strings* `instruction_recall_dataset/build_instruction_dataset.py` uses for
every real example (see `format_agnews` / `format_mnli` there), with only
the example-specific content field(s) left empty. This keeps the constant
part of the template -- the part that plausibly creates the label-string
prior the raw evaluator is bailing out on -- identical between the real and
null prompts, isolating exactly the content-independent contribution to
each label's loss. Not an invented prompt: it's the real template with an
empty content slot, nothing else changed.

  AG News null prompt:
    "Instruction: Classify the topic of the news article. Answer with one
     of: World, Sports, Business, Sci/Tech.\n\nText: \nLabel:"

  MNLI null prompt:
    "Instruction: Determine the relationship between the premise and the
     hypothesis. Answer with one of: entailment, neutral, contradiction.
     \n\nPremise: \nHypothesis: \nLabel:"
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from typing import Any

import torch

sys.path.insert(0, "/home/noora/dp-llm")
sys.path.insert(0, "/home/noora/dp-llm/evaluation")
from loss_based_eval import _avg_label_loss, _build_prompt, _LABELS_BY_TASK  # noqa: E402


NULL_INPUT_BY_TASK: dict[str, str] = {
    "classification": (
        "Instruction: Classify the topic of the news article. "
        "Answer with one of: World, Sports, Business, Sci/Tech.\n\n"
        "Text: \n"
        "Label:"
    ),
    "nli": (
        "Instruction: Determine the relationship between the premise and the hypothesis. "
        "Answer with one of: entailment, neutral, contradiction.\n\n"
        "Premise: \n"
        "Hypothesis: \n"
        "Label:"
    ),
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def build_stratified_sample(val_path: str, per_class: int = 20, seed: int = 42) -> list[dict]:
    """Fixed stratified sample: >= per_class examples for every AG News and
    every MNLI label, deterministically shuffled (seed=42) within each
    class before truncating, so the sample isn't just "whatever came first
    in the file" but is still fully reproducible."""
    by_task_label: dict[tuple[str, str], list[dict]] = {}
    with open(val_path) as f:
        for line in f:
            rec = json.loads(line)
            task = rec.get("task")
            if task not in _LABELS_BY_TASK:
                continue
            key = (task, rec["target"])
            by_task_label.setdefault(key, []).append(rec)

    rng = random.Random(seed)
    sample = []
    for key, records in sorted(by_task_label.items()):
        records = list(records)
        rng.shuffle(records)
        sample.extend(records[:per_class])
    return sample


@torch.no_grad()
def raw_and_calibrated_predictions(
    model, tokenizer, device, task: str, input_text: str,
    null_losses: dict[str, float], max_length: int = 512,
) -> tuple[str, str, dict[str, float]]:
    labels = _LABELS_BY_TASK[task]
    prompt = _build_prompt(tokenizer, input_text)
    losses = {
        label: _avg_label_loss(model, tokenizer, device, prompt, label, max_length=max_length)
        for label in labels
    }
    raw_pred = min(losses, key=losses.get)
    calibrated = {label: losses[label] - null_losses[label] for label in labels}
    cal_pred = min(calibrated, key=calibrated.get)
    return raw_pred, cal_pred, losses


@torch.no_grad()
def compute_null_losses(model, tokenizer, device, task: str, max_length: int = 512) -> dict[str, float]:
    labels = _LABELS_BY_TASK[task]
    prompt = _build_prompt(tokenizer, NULL_INPUT_BY_TASK[task])
    return {
        label: _avg_label_loss(model, tokenizer, device, prompt, label, max_length=max_length)
        for label in labels
    }


@torch.no_grad()
def generation_prediction(
    model, tokenizer, device, task: str, input_text: str, max_new_tokens: int = 8,
) -> tuple[str, str]:
    """Returns (prediction_or_'invalid', raw_decoded_text).

    Uses the exact same `_build_prompt` as the raw/calibrated methods (and
    as training), not `canarybench.prompt_format.build_generation_prompt` --
    that function returns the bare input with no "### Response:" suffix for
    non-chat-template tokenizers, which would silently make this method's
    prompt inconsistent with the other two and with what the model was
    actually trained on."""
    prompt = _build_prompt(tokenizer, input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    first_line = decoded.split("\n")[0]
    normalized = _norm(first_line)
    # exact match only, after case/whitespace normalization -- no fuzzy matching
    label_lookup = {_norm(l): l for l in _LABELS_BY_TASK[task]}
    if normalized in label_lookup:
        return label_lookup[normalized], decoded
    return "invalid", decoded


def confusion_matrix(golds: list[str], preds: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    all_cols = labels + ["invalid"]
    cm = {g: {c: 0 for c in all_cols} for g in labels}
    for g, p in zip(golds, preds):
        col = p if p in all_cols else "invalid"
        cm[g][col] += 1
    return cm


def summarize(task: str, golds: list[str], preds: list[str]) -> dict[str, Any]:
    labels = _LABELS_BY_TASK[task]
    n = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    invalid = sum(1 for p in preds if p == "invalid")
    dist = dict(Counter(preds))
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "invalid_rate": round(invalid / n, 4) if n else None,
        "prediction_distribution": dist,
        "confusion_matrix": confusion_matrix(golds, preds, labels),
        "all_labels_predicted": all(l in dist for l in labels),
    }

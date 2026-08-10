"""
Loss-based evaluation for base (non-instruction) causal LMs on classification-style tasks.

For tasks like topic classification or NLI, generation is unreliable for base models
(especially DP-trained). Instead, compute the loss of each candidate label as a
continuation of the prompt and pick the lowest-loss label.
"""

from __future__ import annotations

from typing import Any

import torch


# Tasks that should use loss-based label selection (not generation).
TASK_LABELS: set[str] = {"classification", "nli"}

# Candidate labels per task (must match dataset `target` strings).
_LABELS_BY_TASK: dict[str, list[str]] = {
    "classification": ["World", "Sports", "Business", "Sci/Tech"],
    "nli": ["entailment", "contradiction", "neutral"],
}


def _build_prompt(tokenizer, input_text: str) -> str:
    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    if has_chat:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return input_text + "\n\n### Response:\n"


def _avg_label_loss(
    model,
    tokenizer,
    device: str,
    prompt: str,
    label: str,
    *,
    max_length: int = 512,
) -> float:
    """
    Compute average token loss for `label` when appended after `prompt`.
    Only label tokens contribute to the loss; prompt tokens are ignored.
    """
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    full_text = prompt + label
    full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = full["input_ids"].to(device)
    attention_mask = full.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = int(prompt_ids.shape[1])
    labels = input_ids.clone()
    if prompt_len > 0:
        labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(1, -1)

    mask = (shift_labels != -100).float()
    denom = mask.sum().clamp(min=1.0)
    return float((per_token * mask).sum() / denom)


def evaluate_classification_by_loss(
    model,
    tokenizer,
    dataset,
    device: str,
    *,
    max_samples: int | None = None,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """
    Return samples in the same schema as generation-based evaluation:
    {group, task, input, target, generated}

    Only records whose `task` is in TASK_LABELS are evaluated; others are skipped.
    """
    model.eval()
    model.to(device)

    n_total = len(dataset.records)
    if max_samples is not None and max_samples > 0:
        n_total = min(n_total, max_samples)

    samples: list[dict[str, Any]] = []

    for rec in dataset.records[:n_total]:
        task = rec.get("task", "unknown")
        if task not in TASK_LABELS:
            continue

        labels = _LABELS_BY_TASK.get(task)
        if not labels:
            continue

        prompt = _build_prompt(tokenizer, rec["input"])

        best_label = None
        best_loss = None
        for label in labels:
            loss = _avg_label_loss(
                model,
                tokenizer,
                device,
                prompt,
                label,
                max_length=max_length,
            )
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_label = label

        samples.append(
            {
                "group": rec.get("group", "unknown"),
                "task": task,
                "input": rec["input"],
                "target": rec["target"],
                "generated": (best_label or "").strip(),
            }
        )

    return samples


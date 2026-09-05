"""Canonical prompt/target formatting, shared by training and evaluation.

Reference behavior for this module is what `finetune/dataset.py`
(`MultiTaskDataset._format_record`, the actual training-data formatter) and
`metrics/base.py` (`build_supervised_text`, the canonical canary-exposure
evaluation path used by `benchmarks/cross_model_exposure.py`) already did
correctly and identically before this refactor: prefer the tokenizer's own
`chat_template` via `apply_chat_template`, and only fall back to a hardcoded
"### Response:" template when the tokenizer has no chat template at all.

Both call sites are now expected to import from here instead of keeping
their own copies, so training and evaluation cannot silently drift apart.

`legacy_hardcoded_supervised_text()` / `legacy_hardcoded_prompt()` preserve,
byte-for-byte, the OLD unconditional "### Response:" formatting that
`finetune/canary_eval.py`, `benchmarks/finprivcanary/canary_evaluator.py`,
and `benchmarks/finprivcanary/crash_safe_checkpoint.py` used previously.
That old formatting never fed the training loss itself (see
`finetune/dataset.py`) -- it only affected the per-epoch canary-loss
*logging* during training and a few standalone/legacy evaluation helpers.
It is kept here, clearly labeled, so historical logs/results produced with
it can still be reproduced exactly on request.
"""

import os
from typing import Any, Dict, Tuple

RESPONSE_PREFIX = "\n\n### Response:\n"


def _chat_template_kwargs() -> Dict[str, str]:
    """Optional reproducibility override for date-aware chat templates."""
    date_string = os.environ.get("DP_LLM_CHAT_DATE")
    return {"date_string": date_string} if date_string else {}


def tokenizer_uses_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_generation_prompt(tokenizer: Any, prompt: str) -> str:
    """Prompt-only string used to *generate* a completion (no target).

    This must be identical to the prompt-only prefix returned by
    :func:`build_supervised_text` for a non-LM record. In particular,
    tokenizers without a chat template (VaultGemma) use RESPONSE_PREFIX;
    returning the raw prompt here would make greedy evaluation use a
    different format from training.
    """
    if tokenizer_uses_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(),
        )
    return f"{prompt}{RESPONSE_PREFIX}"


def build_supervised_text(tokenizer: Any, record: Dict[str, Any]) -> Tuple[str, str]:
    """Canonical (prompt_only, full_text) pair for a training/eval record.

    `record` must have "input"/"target" string fields and may have a
    "task" field; records with task == "lm" (or empty/whitespace-only
    "input") are treated as pure language-modeling text with no prompt,
    matching `finetune/dataset.py`.
    """
    prompt = record.get("input", "")
    target = record.get("target", "")
    if record.get("task") == "lm" or not prompt.strip():
        return "", target
    if tokenizer_uses_chat_template(tokenizer):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            tokenize=False,
            add_generation_prompt=False,
            **_chat_template_kwargs(),
        )
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(),
        )
        return prompt_only, full_text
    prompt_only = f"{prompt}{RESPONSE_PREFIX}"
    full_text = f"{prompt_only}{target}"
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and not full_text.endswith(eos_token):
        full_text += eos_token
    return prompt_only, full_text


def legacy_hardcoded_prompt(record: Dict[str, Any]) -> str:
    """Byte-for-byte the old unconditional template: input + "### Response:\n".

    No chat-template awareness. Use only to reproduce historical
    logs/results that were generated with this formatting.
    """
    return record.get("input", "") + RESPONSE_PREFIX


def legacy_hardcoded_supervised_text(record: Dict[str, Any]) -> Tuple[str, str]:
    """Legacy (prompt_only, full_text) pair matching the old canary_eval.py /
    canary_evaluator.py / crash_safe_checkpoint.py behavior exactly."""
    prompt_only = legacy_hardcoded_prompt(record)
    return prompt_only, prompt_only + record.get("target", "")

import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Canonical prompt/target formatting now lives in canarybench.prompt_format,
# shared with finetune/dataset.py (the training-data formatter). Re-exported
# here under the original names so existing `from metrics.base import ...`
# call sites keep working unchanged.
from canarybench.prompt_format import (  # noqa: F401
    RESPONSE_PREFIX,
    build_generation_prompt,
    build_supervised_text,
    tokenizer_uses_chat_template,
)


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def save_json(path: str, payload: Dict[str, Any]) -> str:
    import os

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def format_supervised_text(record: Dict[str, Any]) -> str:
    if record.get("task") == "lm" or not record.get("input", "").strip():
        return record.get("target", "")
    return f"{record.get('input', '')}{RESPONSE_PREFIX}{record.get('target', '')}"


def encode_prompt_and_target(
    tokenizer: Any,
    record: Dict[str, Any],
    max_length: int = 512,
    device: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt, full_text = build_supervised_text(tokenizer, record)
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded, prompt_len


def compute_sequence_loss(
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    device: str,
    max_length: int = 512,
    mask_prompt: bool = False,
) -> float:
    encoded, prompt_len = encode_prompt_and_target(
        tokenizer=tokenizer,
        record=record,
        max_length=max_length,
        device=device,
    )
    labels = encoded["input_ids"].clone()
    if mask_prompt:
        labels[:, :prompt_len] = -100
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    with torch.no_grad():
        outputs = model(**encoded, labels=labels)
    return float(outputs.loss.item())


def compute_token_losses(
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    device: str,
    max_length: int = 512,
    mask_prompt: bool = True,
) -> np.ndarray:
    encoded, prompt_len = encode_prompt_and_target(
        tokenizer=tokenizer,
        record=record,
        max_length=max_length,
        device=device,
    )
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits[:, :-1, :].float()
    targets = encoded["input_ids"][:, 1:]
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape[0], targets.shape[1])[0]
    if mask_prompt:
        prompt_token_losses = max(prompt_len - 1, 0)
        token_losses = token_losses[prompt_token_losses:]
    return token_losses.detach().cpu().numpy()


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_input_length: int = 256,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    already_formatted: bool = False,
) -> str:
    """already_formatted=True skips build_generation_prompt's own chat-template
    wrapping, for callers that already built the exact training-consistent
    prompt themselves (e.g. via canarybench.prompt_format.build_supervised_text's
    prompt_only) and would otherwise get it wrapped a second time."""
    if not already_formatted:
        prompt = build_generation_prompt(tokenizer, prompt)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_first_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("### Response:"):
        cleaned = cleaned[len("### Response:"):].strip()
    if "\n\n### Response:" in cleaned:
        cleaned = cleaned.split("\n\n### Response:", 1)[0].strip()
    if "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[0].strip()
    return cleaned


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(lowered.split())


def tokenize_normalized(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def qa_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_normalized(prediction)
    ref_tokens = tokenize_normalized(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def rouge1_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_normalized(prediction)
    ref_tokens = tokenize_normalized(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def macro_f1(predictions: Sequence[str], references: Sequence[str]) -> Optional[float]:
    if not predictions or not references:
        return None
    labels = sorted(set(predictions) | set(references))
    if not labels:
        return None
    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref == label)
        fp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref != label)
        fn = sum(1 for pred, ref in zip(predictions, references) if pred != label and ref == label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(predictions: Sequence[str], references: Sequence[str]) -> Optional[float]:
    if not predictions or not references:
        return None
    return float(sum(pred == ref for pred, ref in zip(predictions, references)) / len(predictions))


def exp_perplexity(avg_loss: float) -> float:
    try:
        return float(math.exp(avg_loss))
    except OverflowError:
        return float("inf")


def load_causal_lm(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: Optional[Any] = torch.bfloat16,
) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Always load the tokenizer from the base model repo/path.
    # Some adapter output dirs may contain a `tokenizer.json` / `tokenizer_config.json`
    # with an incompatible/placeholder `tokenizer_class` (e.g. "TokenizersBackend"),
    # which breaks AutoTokenizer loading.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer)

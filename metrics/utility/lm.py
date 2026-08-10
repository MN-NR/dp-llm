from typing import Any, Dict, Optional

import torch

from metrics.base import (
    encode_prompt_and_target,
    exp_perplexity,
    generate_completion,
    load_jsonl,
    rouge1_f1,
)


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    data_path: str,
    device: str,
    max_samples: int = 500,
    group: Optional[str] = None,
    max_length: int = 512,
) -> Dict[str, Optional[float]]:
    records = load_jsonl(data_path)
    if group is not None:
        records = [record for record in records if record.get("group") == group]
    records = records[:max_samples]

    total_loss = 0.0
    total_tokens = 0
    for record in records:
        encoded, prompt_len = encode_prompt_and_target(
            tokenizer=tokenizer,
            record=record,
            max_length=max_length,
            device=device,
        )
        labels = encoded["input_ids"].clone()
        labels[:, :prompt_len] = -100
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        token_count = int((labels != -100).sum().item())
        if token_count == 0:
            continue
        with torch.no_grad():
            outputs = model(**encoded, labels=labels)
        if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
            continue
        total_loss += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    avg_loss = (total_loss / total_tokens) if total_tokens else None
    return {
        "perplexity": round(exp_perplexity(avg_loss), 4) if avg_loss is not None else None,
        "avg_loss": round(avg_loss, 4) if avg_loss is not None else None,
        "group": group,
        "n_samples": len(records),
        "n_tokens": total_tokens,
    }


# def group_c_reconstruction(
#     model: Any,
#     tokenizer: Any,
#     val_path: str,
#     device: str,
#     max_samples: int = 200,
#     max_new_tokens: int = 100,
# ) -> Dict[str, Optional[float]]:
#     records = [record for record in load_jsonl(val_path) if record.get("group") == "C"][:max_samples]
#     scores = []
#     for record in records:
#         prediction = generate_completion(
#             model=model,
#             tokenizer=tokenizer,
#             prompt=record.get("input", ""),
#             device=device,
#             max_new_tokens=max_new_tokens,
#         )
#         scores.append(rouge1_f1(prediction, record.get("target", "")))

#     return {
#         "group_c_reconstruction_rouge1": round(sum(scores) / len(scores), 4) if scores else None,
#         "num_evaluated": len(scores),
#     }

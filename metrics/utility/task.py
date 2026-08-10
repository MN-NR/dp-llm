from typing import Any, Dict, List, Optional

from metrics.base import (
    accuracy,
    extract_first_response,
    generate_completion,
    load_jsonl,
    macro_f1,
    qa_f1_score,
    rouge1_f1,
    safe_mean,
)


def canonicalize_label(prediction: str, candidates: List[str]) -> str:
    normalized_prediction = prediction.strip().lower()
    if not normalized_prediction:
        return normalized_prediction
    for candidate in candidates:
        candidate_norm = candidate.strip().lower()
        if normalized_prediction == candidate_norm:
            return candidate_norm
    for candidate in candidates:
        candidate_norm = candidate.strip().lower()
        if normalized_prediction.startswith(candidate_norm):
            return candidate_norm
    for candidate in candidates:
        candidate_norm = candidate.strip().lower()
        if candidate_norm in normalized_prediction.split():
            return candidate_norm
    return normalized_prediction


def evaluate_group_a(
    model: Any,
    tokenizer: Any,
    val_path: str,
    device: str,
    max_samples: int = 500,
    max_new_tokens: int = 50,
) -> Dict[str, Optional[float]]:
    records = [record for record in load_jsonl(val_path) if record.get("group") == "A"][:max_samples]
    classification_labels = sorted({record["target"].strip().lower() for record in records if record.get("task") == "classification"})
    nli_labels = sorted({record["target"].strip().lower() for record in records if record.get("task") == "nli"})

    classification_preds: List[str] = []
    classification_refs: List[str] = []
    nli_preds: List[str] = []
    nli_refs: List[str] = []
    qa_scores: List[float] = []
    summarization_scores: List[float] = []

    for record in records:
        raw_prediction = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=record.get("input", ""),
            device=device,
            max_new_tokens=max_new_tokens,
        ).strip()
        prediction = extract_first_response(raw_prediction)
        target = record.get("target", "").strip()
        task = record.get("task", "unknown")

        if task == "classification":
            classification_preds.append(canonicalize_label(prediction, classification_labels))
            classification_refs.append(target.lower())
        elif task == "nli":
            nli_preds.append(canonicalize_label(prediction, nli_labels))
            nli_refs.append(target.lower())
        elif task == "qa":
            qa_scores.append(qa_f1_score(prediction, target))
        elif task == "summarization":
            summarization_scores.append(rouge1_f1(prediction, target))

    return {
        "classification_acc": round(accuracy(classification_preds, classification_refs), 4)
        if classification_preds
        else None,
        "nli_f1": round(macro_f1(nli_preds, nli_refs), 4) if nli_preds else None,
        "qa_f1": round(safe_mean(qa_scores), 4) if qa_scores else None,
        "summarization_rouge1": round(safe_mean(summarization_scores), 4)
        if summarization_scores
        else None,
        "num_group_a_examples": len(records),
    }


class GroupAUtility:
    def compute(self, model: Any, tokenizer: Any, eval_path: str, device: str, max_samples: int = 500) -> Dict[str, Optional[float]]:
        return evaluate_group_a(
            model=model,
            tokenizer=tokenizer,
            val_path=eval_path,
            device=device,
            max_samples=max_samples,
        )

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, Optional[float]]:
        scores = [rouge1_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
        return {"rouge1": safe_mean(scores)}

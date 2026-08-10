import math
import re
import string
from collections import Counter


try:
    from rouge_score import rouge_scorer
except Exception as e:  # pragma: no cover
    rouge_scorer = None
    _ROUGE_IMPORT_ERROR = e

try:
    from sklearn.metrics import roc_auc_score
except Exception as e:  # pragma: no cover
    roc_auc_score = None
    _SKLEARN_IMPORT_ERROR = e


# ── Text normalization ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, extra spaces."""
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


# ── Classification / NLI ─────────────────────────────────────────────────────

def classification_accuracy(predictions, targets) -> float:
    """
    Exact match after normalization.
    Works for both classification and NLI.
    """
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_word = normalize(pred).split()[0] if normalize(pred) else ""
        target_word = normalize(target).split()[0] if normalize(target) else ""
        if pred_word == target_word:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


# ── QA — F1 + Exact Match (SQuAD style) ──────────────────────────────────────

def qa_f1(prediction, target) -> float:
    pred_tokens = normalize(prediction).split()
    target_tokens = normalize(target).split()

    if not pred_tokens or not target_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_exact_match(prediction, target) -> int:
    return int(normalize(prediction) == normalize(target))


def evaluate_qa(predictions, targets):
    f1_scores = [qa_f1(p, t) for p, t in zip(predictions, targets)]
    em_scores = [qa_exact_match(p, t) for p, t in zip(predictions, targets)]
    denom = len(f1_scores) if f1_scores else 1
    return {
        "f1": sum(f1_scores) / denom,
        "exact_match": sum(em_scores) / denom,
    }


# ── ROUGE for summarization + span corrupt ────────────────────────────────────

def evaluate_rouge(predictions, targets):
    if rouge_scorer is None:  # pragma: no cover
        raise ImportError(
            "Missing dependency for ROUGE: `rouge-score`. "
            f"Original import error: {_ROUGE_IMPORT_ERROR}"
        )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []

    for pred, target in zip(predictions, targets):
        scores = scorer.score(target, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)

    denom = len(r1) if r1 else 1
    return {
        "rouge1": sum(r1) / denom,
        "rouge2": sum(r2) / denom,
        "rougeL": sum(rL) / denom,
    }


# ── PII masking — entity-level F1 ────────────────────────────────────────────

def extract_entities(text: str):
    """Extract all [ENTITY_TYPE] tokens from text."""
    return re.findall(r"\[[A-Z_]+\]", text or "")


def pii_mask_f1(prediction, target):
    pred_entities = Counter(extract_entities(prediction))
    target_entities = Counter(extract_entities(target))

    tp = sum((pred_entities & target_entities).values())
    fp = sum((pred_entities - target_entities).values())
    fn = sum((target_entities - pred_entities).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_pii_mask(predictions, targets):
    results = [pii_mask_f1(p, t) for p, t in zip(predictions, targets)]
    denom = len(results) if results else 1
    return {
        "precision": sum(r["precision"] for r in results) / denom,
        "recall": sum(r["recall"] for r in results) / denom,
        "f1": sum(r["f1"] for r in results) / denom,
        "extraction_rate": sum(r["recall"] for r in results) / denom,
    }


# ── PII NER — exact match on sorted entity list ───────────────────────────────

def evaluate_pii_ner(predictions, targets):
    """
    Target: "ADDRESS, EMAIL, NAME"
    Pred:   "EMAIL, NAME, ADDRESS"  ← same but different order → correct
    """
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_labels = sorted(set(normalize(pred).split(",")))
        target_labels = sorted(set(normalize(target).split(",")))
        pred_labels = [l.strip() for l in pred_labels if l.strip()]
        target_labels = [l.strip() for l in target_labels if l.strip()]
        if pred_labels == target_labels:
            correct += 1
    return {"exact_match": correct / len(predictions) if predictions else 0.0}


# ── PII detection — set overlap on ';' delimited values ───────────────────────

def _split_pii_values(text):
    if text is None:
        return set()
    return {x.strip().lower() for x in str(text).split(";") if x.strip()}


def evaluate_pii_detection(samples):
    tp = fp = fn = exact = 0

    for s in samples:
        pred = _split_pii_values(s.get("generated", ""))
        gold = _split_pii_values(s.get("target", ""))

        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

        if pred == gold:
            exact += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = exact / len(samples) if samples else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(exact_match, 4),
        "n_samples": len(samples),
    }

# ── Privacy metrics ───────────────────────────────────────────────────────────

def canary_exposure(member_losses, reference_losses):
    """
    Carlini et al. exposure metric.
    For each member canary: what fraction of reference canaries
    have lower loss? → rank → exposure = -log2(rank/N)

    Higher exposure = more memorization = worse privacy
    """
    exposures = []
    n_ref = len(reference_losses)
    if n_ref == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "values": []}

    for member_loss in member_losses:
        rank = sum(1 for ref_loss in reference_losses if ref_loss < member_loss)
        rank = max(rank, 1)
        exposures.append(-math.log2(rank / n_ref))

    return {
        "mean": sum(exposures) / len(exposures) if exposures else 0.0,
        "max": max(exposures) if exposures else 0.0,
        "min": min(exposures) if exposures else 0.0,
        "values": exposures,
    }


def frequency_aware_exposure(exposure_scores, frequencies):
    """Normalize exposure by log frequency so canaries seen 1x vs 10x are comparable."""
    normalized = []
    for exp, freq in zip(exposure_scores, frequencies):
        norm_exp = exp / math.log2(freq + 1) if freq > 0 else exp
        normalized.append(norm_exp)
    return {
        "mean": sum(normalized) / len(normalized) if normalized else 0.0,
        "values": normalized,
    }


def mia_auc(member_losses, nonmember_losses) -> float:
    """
    Membership Inference Attack AUC.
    Score = -loss (lower loss = more likely member)
    AUC = 0.5 → random = perfect privacy
    AUC = 1.0 → perfect attack = no privacy
    """
    if roc_auc_score is None:  # pragma: no cover
        raise ImportError(
            "Missing dependency for MIA AUC: `scikit-learn`. "
            f"Original import error: {_SKLEARN_IMPORT_ERROR}"
        )
    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)
    scores = [-l for l in member_losses] + [-l for l in nonmember_losses]
    return roc_auc_score(labels, scores)


# ── Master evaluation function ────────────────────────────────────────────────

def compute_all_metrics(samples_by_task: dict) -> dict:
    """
    Input:
        samples_by_task = {
            "classification": [{"generated": "...", "target": "..."}, ...],
            "nli":            [...],
            "qa":             [...],
            ...
        }

    Output: dict of all metrics
    """
    metrics = {}

    if "classification" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["classification"]]
        targets = [s["target"] for s in samples_by_task["classification"]]
        metrics["classification"] = {
            "accuracy": classification_accuracy(preds, targets),
            "n_samples": len(preds),
        }

    if "nli" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["nli"]]
        targets = [s["target"] for s in samples_by_task["nli"]]
        metrics["nli"] = {
            "accuracy": classification_accuracy(preds, targets),
            "n_samples": len(preds),
        }

    if "qa" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["qa"]]
        targets = [s["target"] for s in samples_by_task["qa"]]
        metrics["qa"] = {**evaluate_qa(preds, targets), "n_samples": len(preds)}

    if "summarization" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["summarization"]]
        targets = [s["target"] for s in samples_by_task["summarization"]]
        metrics["summarization"] = {**evaluate_rouge(preds, targets), "n_samples": len(preds)}

    if "span_corrupt" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["span_corrupt"]]
        targets = [s["target"] for s in samples_by_task["span_corrupt"]]
        metrics["span_corrupt"] = {**evaluate_rouge(preds, targets), "n_samples": len(preds)}

    if "pii_mask" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["pii_mask"]]
        targets = [s["target"] for s in samples_by_task["pii_mask"]]
        metrics["pii_mask"] = {**evaluate_pii_mask(preds, targets), "n_samples": len(preds)}

    if "pii_ner" in samples_by_task:
        preds = [s["generated"] for s in samples_by_task["pii_ner"]]
        targets = [s["target"] for s in samples_by_task["pii_ner"]]
        metrics["pii_ner"] = {**evaluate_pii_ner(preds, targets), "n_samples": len(preds)}

    if "pii_detection" in samples_by_task:
        metrics["pii_detection"] = evaluate_pii_detection(samples_by_task["pii_detection"])

    return metrics


def print_metrics_table(metrics: dict, model_key: str, epsilon):
    """Print a clean table for paper."""
    print(f"\n{'=' * 65}")
    print(f"METRICS — {model_key}  ε={epsilon}")
    print(f"{'=' * 65}")

    if "classification" in metrics:
        print(f"  Classification Accuracy : {metrics['classification']['accuracy']:.4f}")
    if "nli" in metrics:
        print(f"  NLI Accuracy            : {metrics['nli']['accuracy']:.4f}")
    if "qa" in metrics:
        print(f"  QA F1                   : {metrics['qa']['f1']:.4f}")
        print(f"  QA Exact Match          : {metrics['qa']['exact_match']:.4f}")
    if "summarization" in metrics:
        print(f"  Summarization ROUGE-1   : {metrics['summarization']['rouge1']:.4f}")
        print(f"  Summarization ROUGE-L   : {metrics['summarization']['rougeL']:.4f}")
    if "span_corrupt" in metrics:
        print(f"  Span Corrupt ROUGE-L    : {metrics['span_corrupt']['rougeL']:.4f}")
    if "pii_mask" in metrics:
        print(f"  PII Mask Precision      : {metrics['pii_mask']['precision']:.4f}")
        print(f"  PII Mask Recall         : {metrics['pii_mask']['recall']:.4f}")
        print(f"  PII Mask F1             : {metrics['pii_mask']['f1']:.4f}")
        print(f"  PII Extraction Rate     : {metrics['pii_mask']['extraction_rate']:.4f}")
    if "pii_ner" in metrics:
        print(f"  PII NER Exact Match     : {metrics['pii_ner']['exact_match']:.4f}")
    if "pii_detection" in metrics:
        print(f"  PII Detect Precision    : {metrics['pii_detection']['precision']:.4f}")
        print(f"  PII Detect Recall       : {metrics['pii_detection']['recall']:.4f}")
        print(f"  PII Detect F1           : {metrics['pii_detection']['f1']:.4f}")
        print(f"  PII Detect Exact Match  : {metrics['pii_detection']['exact_match']:.4f}")
    print(f"{'=' * 65}")

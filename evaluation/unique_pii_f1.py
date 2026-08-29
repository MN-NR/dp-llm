import json
import re
import sys


def _norm_ws(x):
    return re.sub(r"\s+", " ", (x or "").lower().strip())


def _split_pii_relaxed(text):
    return [_norm_ws(x) for x in re.split(r"[;\n]", str(text or "")) if x.strip()]


def _relaxed_match(pred, gold, overlap_threshold=0.5):
    pred = _norm_ws(pred)
    gold = _norm_ws(gold)
    if pred == gold:
        return True
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    if not pred_tokens or not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)
    return overlap >= overlap_threshold


def unique_relaxed_pii_scores(samples, overlap_threshold=0.5):
    """Same matching rule as relaxed_pii_scores (test_model.py), but predicted
    AND gold items are deduplicated (as sets) before matching, isolating
    whether duplicate predictions -- not the looser overlap matching itself --
    explain the gap between strict and relaxed F1."""
    tp = fp = fn = exact = 0

    total_pred_items_raw = 0
    total_pred_items_unique = 0
    per_example_dup_counts = []
    examples_with_dup = 0

    for s in samples:
        golds_raw = _split_pii_relaxed(s.get("target", ""))
        preds_raw = _split_pii_relaxed(s.get("generated", ""))

        # dedup, order-preserving
        golds = list(dict.fromkeys(golds_raw))
        preds = list(dict.fromkeys(preds_raw))

        n_raw = len(preds_raw)
        n_unique = len(preds)
        dup_count = n_raw - n_unique
        total_pred_items_raw += n_raw
        total_pred_items_unique += n_unique
        per_example_dup_counts.append(dup_count)
        if dup_count > 0:
            examples_with_dup += 1

        matched_gold = set()
        matched_pred = set()
        for pi, p in enumerate(preds):
            for gi, g in enumerate(golds):
                if gi in matched_gold:
                    continue
                if _relaxed_match(p, g, overlap_threshold=overlap_threshold):
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        tp += len(matched_gold)
        fp += len(preds) - len(matched_pred)
        fn += len(golds) - len(matched_gold)

        if set(preds) == set(golds):
            exact += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n = len(samples)
    repetition_fraction = (
        (total_pred_items_raw - total_pred_items_unique) / total_pred_items_raw
        if total_pred_items_raw else 0.0
    )
    mean_duplicate_count = sum(per_example_dup_counts) / n if n else 0.0

    return {
        "unique_precision": round(precision, 4),
        "unique_recall": round(recall, 4),
        "unique_f1": round(f1, 4),
        "unique_exact_match": round((exact / n) if n else 0.0, 4),
        "repetition_fraction": round(repetition_fraction, 4),
        "mean_duplicate_count": round(mean_duplicate_count, 4),
        "examples_with_duplicates": examples_with_dup,
        "n_samples": n,
    }


def main(path):
    d = json.load(open(path))
    for ev in d["evaluations"]:
        if ev["dataset"] != "val":
            continue
        samples = ev.get("samples", [])
        pii_samples = [s for s in samples if s.get("task") == "pii_detection"]
        print(f"file: {path}")
        print(f"  total samples saved: {len(samples)}  pii_detection samples: {len(pii_samples)}")
        if not pii_samples:
            print("  NO pii_detection samples found -- was --save_all_samples passed?")
            return
        result = unique_relaxed_pii_scores(pii_samples)
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1])

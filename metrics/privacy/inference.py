from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from metrics.base import compute_sequence_loss, compute_token_losses


DEFAULT_WBC_WINDOW_LENGTHS = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]


def _window_vote_score(
    target_losses: Sequence[float],
    reference_losses: Sequence[float],
    window_size: int,
) -> float:
    min_length = min(len(target_losses), len(reference_losses))
    if min_length == 0:
        return 0.0
    effective_window = min(window_size, min_length)
    kernel = np.ones(effective_window)
    target_sums = np.convolve(np.asarray(target_losses[:min_length]), kernel, mode="valid")
    reference_sums = np.convolve(np.asarray(reference_losses[:min_length]), kernel, mode="valid")
    return float(np.mean(reference_sums > target_sums))


def _loss_attack_score(sequence_loss: float) -> float:
    return -float(sequence_loss)


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for index, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += index
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _roc_curve(labels: Sequence[int], scores: Sequence[float]) -> List[Dict[str, float]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    curve = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        curve.append(
            {
                "threshold": float(score),
                "fpr": fp / negatives if negatives else 0.0,
                "tpr": tp / positives if positives else 0.0,
            }
        )
    return curve


def tpr_at_fpr(curve: Sequence[Dict[str, float]], target_fpr: float) -> float:
    eligible = [point["tpr"] for point in curve if point["fpr"] <= target_fpr]
    return float(max(eligible)) if eligible else 0.0


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_bootstrap_samples: int = 200,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    n = len(labels_arr)
    aucs: List[float] = []
    for _ in range(n_bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample_labels = labels_arr[indices]
        if len(set(sample_labels.tolist())) < 2:
            continue
        sample_scores = scores_arr[indices]
        aucs.append(_roc_auc(sample_labels.tolist(), sample_scores.tolist()))
    if not aucs:
        return {"auc_ci_lower": 0.5, "auc_ci_upper": 0.5}
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return {
        "auc_ci_lower": float(np.percentile(aucs, lower_q)),
        "auc_ci_upper": float(np.percentile(aucs, upper_q)),
    }


def run_wbc_attack(
    model: Any,
    tokenizer: Any,
    member_canaries: Sequence[Dict[str, Any]],
    reference_canaries: Sequence[Dict[str, Any]],
    device: str,
    reference_model: Optional[Any] = None,
    reference_tokenizer: Optional[Any] = None,
    require_reference_model: bool = False,
    window_lengths: Optional[Sequence[int]] = None,
    max_length: int = 512,
    n_bootstrap_samples: int = 200,
) -> Dict[str, Any]:
    if require_reference_model and (reference_model is None or reference_tokenizer is None):
        raise ValueError(
            "run_wbc_attack(require_reference_model=True) requires both "
            "`reference_model` and `reference_tokenizer` (e.g., your base/pre-finetune model)."
        )

    reference_model = reference_model or model
    reference_tokenizer = reference_tokenizer or tokenizer
    window_lengths = list(window_lengths or DEFAULT_WBC_WINDOW_LENGTHS)
    uses_reference_model = (reference_model is not model) or (reference_tokenizer is not tokenizer)

    member_scores: List[float] = []
    reference_scores_out: List[float] = []
    member_loss_scores: List[float] = []
    reference_loss_scores: List[float] = []

    for canary in member_canaries:
        target_losses = compute_token_losses(
            model=model,
            tokenizer=tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=False,
        )
        ref_losses = compute_token_losses(
            model=reference_model,
            tokenizer=reference_tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=False,
        )
        window_scores = [_window_vote_score(target_losses, ref_losses, window) for window in window_lengths]
        member_scores.append(float(np.mean(window_scores)) if window_scores else 0.0)
        member_loss_scores.append(
            _loss_attack_score(
                compute_sequence_loss(
                    model=model,
                    tokenizer=tokenizer,
                    record=canary,
                    device=device,
                    max_length=max_length,
                    mask_prompt=False,
                )
            )
        )

    for canary in reference_canaries:
        target_losses = compute_token_losses(
            model=model,
            tokenizer=tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=False,
        )
        ref_losses = compute_token_losses(
            model=reference_model,
            tokenizer=reference_tokenizer,
            record=canary,
            device=device,
            max_length=max_length,
            mask_prompt=False,
        )
        window_scores = [_window_vote_score(target_losses, ref_losses, window) for window in window_lengths]
        reference_scores_out.append(float(np.mean(window_scores)) if window_scores else 0.0)
        reference_loss_scores.append(
            _loss_attack_score(
                compute_sequence_loss(
                    model=model,
                    tokenizer=tokenizer,
                    record=canary,
                    device=device,
                    max_length=max_length,
                    mask_prompt=False,
                )
            )
        )

    labels = [1] * len(member_scores) + [0] * len(reference_scores_out)
    scores = member_scores + reference_scores_out
    curve = _roc_curve(labels, scores)
    auc = _roc_auc(labels, scores)
    loss_auc = _roc_auc(labels, member_loss_scores + reference_loss_scores)

    results = {
        "wbc_auc": round(auc, 4),
        "tpr_at_fpr001": round(tpr_at_fpr(curve, 0.01), 4),
        "window_lengths": window_lengths,
        "window_aggregation": "mean",
        "comparison_rule": "fraction_of_windows_where_reference_loss_exceeds_target_loss",
        "uses_reference_model": uses_reference_model,
        "member_scores": member_scores,
        "reference_scores": reference_scores_out,
        "loss_baseline_auc": round(loss_auc, 4),
        "random_baseline_auc": 0.5,
    }
    results.update(
        {
            key: round(value, 4)
            for key, value in bootstrap_auc_ci(
                labels,
                scores,
                n_bootstrap_samples=n_bootstrap_samples,
            ).items()
        }
    )
    return results


class WindowBasedClassifierAttack:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_length: int = 512,
        reference_model: Optional[Any] = None,
        reference_tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.reference_model = reference_model
        self.reference_tokenizer = reference_tokenizer

    def run(
        self,
        member_canaries: Sequence[Dict[str, Any]],
        reference_canaries: Sequence[Dict[str, Any]],
        window_lengths: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        return run_wbc_attack(
            model=self.model,
            tokenizer=self.tokenizer,
            member_canaries=member_canaries,
            reference_canaries=reference_canaries,
            device=self.device,
            reference_model=self.reference_model,
            reference_tokenizer=self.reference_tokenizer,
            window_lengths=window_lengths,
            max_length=self.max_length,
        )

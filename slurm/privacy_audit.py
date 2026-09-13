#!/usr/bin/env python
"""Privacy audit for ONE selected checkpoint (post checkpoint-selection).
Reports exactly the items in the paper-run spec:
  - actual epsilon (as accumulated through the selected epoch's training
    steps -- DP composition is per-step, not per-checkpoint, so an earlier
    epoch's checkpoint carries a smaller actual epsilon than the final one;
    parsed straight from that epoch's "Epoch N Summary" in train.log)
  - member / reference losses (avg)
  - loss gap (reference - member) + 95% bootstrap CI
  - WBC AUC + 95% bootstrap CI (already computed inside run_wbc_attack)
  - exposure (mean/p50/p95/max)
  - 1x / 10x / 50x repetition-tier exposure breakdown
  - targeted greedy extraction (metrics.privacy.extraction.pii_extraction_rate:
    canonical training-format prompt, greedy decode, exact-substring hit +
    ROUGE-1 against the true canary secret)
Also re-confirms validation utility on the FULL val set (not the subsample
select_checkpoint.py used for ranking) for the report, using include_group_a.
"""
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "slurm"))

import numpy as np  # noqa: E402

from benchmarks.cross_model_exposure import evaluate_checkpoint  # noqa: E402
from metrics.base import save_json  # noqa: E402
from metrics.privacy.memorization import exposure_by_repetition  # noqa: E402
from finetune.config import MODELS  # noqa: E402
# Reuse the EXACT same utility evaluator as checkpoint selection so the
# final reported utility and the selection sweep can't drift apart -- the
# selection sweep runs it on a 300/task subsample, this runs it on full val.
from select_checkpoint import eval_one_checkpoint, load_val_subset  # noqa: E402

DEFAULT_MODEL_KEY = "llama3_2_1b_it"


def bootstrap_gap_ci(member_losses, reference_losses, n_bootstrap_samples=200, alpha=0.95, seed=42):
    """Percentile bootstrap on the (reference_mean - member_mean) loss gap,
    matching metrics.privacy.inference.bootstrap_auc_ci's style/parameters."""
    rng = np.random.default_rng(seed)
    m = np.asarray([float(x["loss"]) for x in member_losses])
    r = np.asarray([float(x["loss"]) for x in reference_losses])
    gaps = []
    for _ in range(n_bootstrap_samples):
        m_s = m[rng.integers(0, len(m), size=len(m))]
        r_s = r[rng.integers(0, len(r), size=len(r))]
        gaps.append(float(r_s.mean() - m_s.mean()))
    lower_q = (1.0 - alpha) / 2.0 * 100.0
    upper_q = (1.0 + alpha) / 2.0 * 100.0
    return {
        "gap_ci_lower": float(np.percentile(gaps, lower_q)),
        "gap_ci_upper": float(np.percentile(gaps, upper_q)),
    }


def epsilon_at_epoch(train_log_path, epoch):
    """Parse the "Epoch N Summary" block's "eps spent" line for the given
    epoch -- this is the ACTUAL epsilon accumulated through that many
    training steps, distinct from the final (epoch-20) epsilon."""
    if not os.path.exists(train_log_path):
        return None
    text = open(train_log_path, encoding="utf-8", errors="replace").read()
    pattern = re.compile(
        rf"Epoch {epoch} Summary.*?eps spent\s*:\s*([\d.]+)", re.IGNORECASE | re.DOTALL
    )
    # Log uses the literal Greek epsilon character; match either spelling.
    pattern2 = re.compile(
        rf"Epoch {epoch} Summary.*?ε spent\s*:\s*([\d.]+)\s*\(target", re.DOTALL
    )
    m = pattern2.search(text) or pattern.search(text)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default=DEFAULT_MODEL_KEY, choices=list(MODELS.keys()))
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--member-path", required=True)
    ap.add_argument("--reference-path", required=True)
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-member-samples", type=int, default=770)
    ap.add_argument("--max-reference-samples", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_model = MODELS[args.model_key].model_name
    adapter_dir = os.path.join(args.ckpt_root, f"epoch_{args.epoch}")
    train_log = os.path.join(args.ckpt_root, "train.log")

    print(f"Auditing {adapter_dir}")
    result = evaluate_checkpoint(
        base_model_path=base_model,
        adapter_path=adapter_dir,
        member_path=args.member_path,
        reference_path=args.reference_path,
        val_path=args.val_path,
        device=args.device,
        reference_model_path=base_model,  # WBC's reference = untuned base model
        max_member_samples=args.max_member_samples,
        max_reference_samples=args.max_reference_samples,
        include_group_a=False,  # do NOT use metrics.utility.task.evaluate_group_a here --
                                # it's a different (fuzzy) evaluator on a different subset
                                # than checkpoint selection. Utility below is recomputed
                                # with the SAME strict evaluator selection uses, on full val.
    )

    # Full-val utility with the exact evaluator used for checkpoint selection
    # (classification_eval_methods.generation_prediction + summarize, and
    # unique_pii_f1) -- max_per_task=0 => all 1000 records/task.
    print("Recomputing full-val utility with the checkpoint-selection evaluator ...", flush=True)
    full_subset = load_val_subset(args.val_path, seed=42, max_per_task=0)
    utility_full = eval_one_checkpoint(
        base_model, adapter_dir,
        full_subset.get("classification", []),
        full_subset.get("nli", []),
        full_subset.get("pii_detection", []),
        args.device,
    )

    exposure = result["canary_exposure"]
    gap_ci = bootstrap_gap_ci(exposure["member_losses"], exposure["reference_losses"])
    freq_breakdown = exposure_by_repetition(exposure["member_losses"], exposure["reference_losses"])
    eps_actual = epsilon_at_epoch(train_log, args.epoch)

    # Targeted greedy extraction rate broken out by canary injection count
    # (1x / 10x / 50x), so extraction can be reported "by frequency" the
    # same way exposure is. Aggregated from the per-canary details.
    ext_by_tier = {}
    _tier_hits = {}
    for d in result["pii_extraction"].get("details", []):
        tier = d.get("exposure")
        if tier is None:
            continue
        tier = int(tier)
        _tier_hits.setdefault(tier, []).append(bool(d.get("exact_hit")))
    for tier, hits in sorted(_tier_hits.items()):
        ext_by_tier[f"extraction_{tier}x"] = {
            "extraction_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "n": len(hits),
        }

    report = {
        "model_key": args.model_key,
        "adapter_path": adapter_dir,
        "selected_epoch": args.epoch,
        "actual_epsilon_at_this_epoch": eps_actual,
        "member_reference_losses": {
            "avg_member_loss": exposure["member_loss_mean"],
            "avg_reference_loss": exposure["reference_loss_mean"],
            "num_members": exposure["num_members"],
            "num_references": exposure["num_references"],
        },
        "gap": {
            "gap": exposure["gap"],
            **gap_ci,
        },
        "exposure": {
            "exposure_mean": exposure["exposure_mean"],
            "exposure_p50": exposure["exposure_p50"],
            "exposure_p95": exposure["exposure_p95"],
            "exposure_max": exposure["exposure_max"],
        },
        "exposure_by_repetition_tier": freq_breakdown,  # keys like "exposure_1", "exposure_10", "exposure_50"
        "wbc_attack": {k: v for k, v in result["wbc"].items() if k not in ("member_scores", "reference_scores")},
        "targeted_greedy_extraction": {
            k: v for k, v in result["pii_extraction"].items() if k != "details"
        },
        "targeted_greedy_extraction_by_frequency": ext_by_tier,
        "utility_full_val": {
            "evaluator": "classification_eval_methods.generation_prediction + summarize "
                         "(same as checkpoint selection); unique_pii_f1 for PII. Full val: "
                         "1000 records/task.",
            "agnews_acc": utility_full["agnews"]["accuracy"],
            "agnews_invalid_rate": utility_full["agnews"]["invalid_rate"],
            "mnli_acc": utility_full["mnli"]["accuracy"],
            "mnli_invalid_rate": utility_full["mnli"]["invalid_rate"],
            "pii_unique_precision": utility_full["pii"]["unique_precision"],
            "pii_unique_recall": utility_full["pii"]["unique_recall"],
            "pii_unique_f1": utility_full["pii"]["unique_f1"],
            "pii_unique_exact_match": utility_full["pii"]["unique_exact_match"],
            "pii_repetition_fraction": utility_full["pii"]["repetition_fraction"],
        },
        "perplexity": result["perplexity"],
        "privacy_composite": result["privacy_composite"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_json(args.out, report)

    print("\n" + "=" * 70)
    print(f"PRIVACY AUDIT -- epoch_{args.epoch}")
    print("=" * 70)
    print(f"  actual epsilon (through this epoch) : {eps_actual}")
    print(f"  avg member loss                     : {exposure['member_loss_mean']}")
    print(f"  avg reference loss                  : {exposure['reference_loss_mean']}")
    print(f"  gap (ref - member)                  : {exposure['gap']:.4f}  "
          f"95% CI [{gap_ci['gap_ci_lower']:.4f}, {gap_ci['gap_ci_upper']:.4f}]")
    print(f"  exposure mean/p50/p95/max           : {exposure['exposure_mean']:.4f} / "
          f"{exposure['exposure_p50']:.4f} / {exposure['exposure_p95']:.4f} / {exposure['exposure_max']:.4f}")
    print(f"  1x/10x/50x exposure                 : {freq_breakdown}")
    wbc = result["wbc"]
    print(f"  WBC AUC                             : {wbc['wbc_auc']}  "
          f"95% CI [{wbc.get('auc_ci_lower')}, {wbc.get('auc_ci_upper')}]  "
          f"(loss-baseline AUC {wbc['loss_baseline_auc']}, random 0.5)")
    ext = result["pii_extraction"]
    print(f"  targeted greedy extraction          : rate={ext['pii_extraction_rate']}  "
          f"avg_rouge1={ext['avg_rouge1']}  n={ext['total_attempts']}")
    if ext_by_tier:
        print(f"  extraction by frequency (1x/10x/50x): "
              + "  ".join(f"{k}={v['extraction_rate']} (n={v['n']})" for k, v in ext_by_tier.items()))
    uf = report["utility_full_val"]
    print(f"  utility (full val, selection evaluator):")
    print(f"    AG News acc={uf['agnews_acc']} invalid={uf['agnews_invalid_rate']}  "
          f"MNLI acc={uf['mnli_acc']} invalid={uf['mnli_invalid_rate']}")
    print(f"    PII unique_f1={uf['pii_unique_f1']} exact_match={uf['pii_unique_exact_match']} "
          f"repetition={uf['pii_repetition_fraction']}")
    print(f"  perplexity                          : {result['perplexity'].get('perplexity')}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

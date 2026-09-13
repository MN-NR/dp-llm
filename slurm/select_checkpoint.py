#!/usr/bin/env python
"""Checkpoint selection: score every saved epoch adapter on VALIDATION
UTILITY ONLY, per the paper-run spec -- no memorization/exposure metric is
computed or consulted here.

Metrics (all greedy decoding, no sampling):
  - AG News  greedy-generation accuracy + invalid-output rate
      (canarybench task "classification"; canonical prompt via
      evaluation.classification_eval_methods.generation_prediction, which
      exact-matches the decoded label against the fixed AG News label set)
  - MNLI     greedy-generation accuracy + invalid-output rate  (task "nli")
  - PII      unique-relaxed F1, unique exact-match, repetition fraction
      (canarybench task "pii_detection"; evaluation.unique_pii_f1's
      duplicate-aware relaxed matcher)

To keep a 20-way sweep tractable, each task is subsampled (fixed seed) to
--max-per-task records for this ranking pass; the checkpoint chosen here is
then re-scored on the FULL validation set as part of the final report (see
privacy_audit.py's utility confirmation), and the full privacy audit runs
on this one selected checkpoint only.

Selection rule (frozen paper spec, not tuned): for each epoch,
  U_val = (agnews_acc + mnli_acc + pii_unique_f1) / 3
Pick argmax U_val. Ties broken by higher pii_unique_f1. Validation utility
ONLY -- no gap / WBC / exposure / extraction / test metric is consulted.
invalid_rate and repetition_fraction are still computed and stored per
epoch for the report, but do NOT enter the selection score.
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

import torch  # noqa: E402

from metrics.base import load_causal_lm, generate_completion  # noqa: E402
from classification_eval_methods import generation_prediction, summarize  # noqa: E402
from loss_based_eval import _build_prompt  # noqa: E402
from unique_pii_f1 import unique_relaxed_pii_scores  # noqa: E402
from finetune.config import MODELS  # noqa: E402

DEFAULT_MODEL_KEY = "llama3_2_1b_it"


def load_val_subset(val_path, seed, max_per_task):
    by_task = {}
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_task.setdefault(rec.get("task"), []).append(rec)
    rng = random.Random(seed)
    subset = {}
    for task, recs in by_task.items():
        recs = list(recs)
        rng.shuffle(recs)
        subset[task] = recs[:max_per_task] if max_per_task else recs
    return subset


def eval_one_checkpoint(base_model, adapter_dir, cls_recs, nli_recs, pii_recs, device):
    loaded = load_causal_lm(base_model, adapter_path=adapter_dir, device=device)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()

    def run_gen_task(task, records):
        preds, golds = [], []
        for r in records:
            pred, _ = generation_prediction(model, tok, device, task, r.get("input", ""))
            preds.append(pred)
            golds.append(r["target"])  # verbatim -- matches _LABELS_BY_TASK casing exactly
        return summarize(task, golds, preds)

    ag = run_gen_task("classification", cls_recs)
    mnli = run_gen_task("nli", nli_recs)

    pii_samples = []
    for r in pii_recs:
        # _build_prompt is chat-template-aware: apply_chat_template for chat
        # models (Llama), "{input}\n\n### Response:\n" fallback for base
        # models (VaultGemma) -- matches training (build_supervised_text) for
        # BOTH, unlike generate_completion's default build_generation_prompt
        # which drops the "### Response:" suffix for non-chat tokenizers.
        prompt = _build_prompt(tok, r.get("input", ""))
        gen = generate_completion(model, tok, prompt=prompt, device=device,
                                  max_new_tokens=80, already_formatted=True)
        pii_samples.append({"target": r.get("target", ""), "generated": gen, "task": "pii_detection"})
    pii = unique_relaxed_pii_scores(pii_samples)

    del model, loaded
    torch.cuda.empty_cache()
    return {"agnews": ag, "mnli": mnli, "pii": pii}


def composite_score(row):
    """U_val = (AG News acc + MNLI acc + PII unique-F1) / 3  -- the frozen
    paper-spec checkpoint-selection score. Validation utility ONLY."""
    ag, mnli, pii = row["agnews"], row["mnli"], row["pii"]
    util = [x for x in (ag["accuracy"], mnli["accuracy"], pii["unique_f1"]) if x is not None]
    return sum(util) / len(util) if util else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default=DEFAULT_MODEL_KEY, choices=list(MODELS.keys()))
    ap.add_argument("--ckpt-root", required=True, help="dir containing epoch_1 .. epoch_N subdirs")
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-per-task", type=int, default=300,
                    help="records per task for the ranking sweep; 0 = full val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-epoch", type=int, default=None,
                    help="evaluate just this one epoch (e.g. re-score the selected "
                         "checkpoint on full val with --max-per-task 0); no selection")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_model = MODELS[args.model_key].model_name
    subset = load_val_subset(args.val_path, args.seed, args.max_per_task)
    cls_recs = subset.get("classification", [])
    nli_recs = subset.get("nli", [])
    pii_recs = subset.get("pii_detection", [])
    print(f"Eval sizes: classification={len(cls_recs)} nli={len(nli_recs)} pii_detection={len(pii_recs)} "
          f"({'FULL VAL' if not args.max_per_task else 'subsample'})")

    epoch_range = [args.only_epoch] if args.only_epoch else list(range(1, args.epochs + 1))
    rows = {}
    for n in epoch_range:
        adapter_dir = os.path.join(args.ckpt_root, f"epoch_{n}")
        if not os.path.isdir(adapter_dir):
            print(f"epoch_{n}: MISSING at {adapter_dir}, skipping")
            continue
        print(f"\n=== epoch_{n} ===", flush=True)
        row = eval_one_checkpoint(base_model, adapter_dir, cls_recs, nli_recs, pii_recs, args.device)
        row["u_val"] = round(composite_score(row), 4)
        rows[n] = row
        ag, mnli, pii = row["agnews"], row["mnli"], row["pii"]
        print(f"  agnews_acc={ag['accuracy']} invalid={ag['invalid_rate']}  "
              f"mnli_acc={mnli['accuracy']} invalid={mnli['invalid_rate']}  "
              f"pii_f1={pii['unique_f1']} pii_em={pii['unique_exact_match']} "
              f"pii_rep={pii['repetition_fraction']}  U_val={row['u_val']}", flush=True)

    if not rows:
        print("No epoch checkpoints found -- nothing to select.")
        return 1

    best_epoch = max(rows, key=lambda n: (rows[n]["u_val"], rows[n]["pii"]["unique_f1"]))

    print("\n" + "=" * 70)
    print("SUMMARY (validation utility only -- no memorization metric used)")
    print("=" * 70)
    print(f"{'epoch':>6} {'agnews':>8} {'ag_inv':>7} {'mnli':>8} {'mnli_inv':>9} "
          f"{'pii_f1':>7} {'pii_em':>7} {'pii_rep':>8} {'U_val':>10}")
    for n in sorted(rows):
        r = rows[n]
        marker = " <== selected" if n == best_epoch else ""
        print(f"{n:>6} {r['agnews']['accuracy']:>8} {r['agnews']['invalid_rate']:>7} "
              f"{r['mnli']['accuracy']:>8} {r['mnli']['invalid_rate']:>9} "
              f"{r['pii']['unique_f1']:>7} {r['pii']['unique_exact_match']:>7} "
              f"{r['pii']['repetition_fraction']:>8} {r['u_val']:>10}{marker}")

    out = {
        "model_key": args.model_key,
        "ckpt_root": args.ckpt_root,
        "selection_rule": (
            "U_val = (agnews_acc + mnli_acc + pii_unique_f1) / 3; "
            "argmax U_val, ties broken by pii_unique_f1. "
            "Validation utility ONLY -- no gap/WBC/exposure/extraction/test metric."
        ),
        "max_per_task": args.max_per_task,
        "seed": args.seed,
        "rows": rows,
        "selected_epoch": best_epoch,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSELECTED: epoch_{best_epoch}  (U_val={rows[best_epoch]['u_val']})")
    print(f"Saved -> {args.out}")
    # Machine-readable line for the calling shell script to grep.
    print(f"SELECTED_EPOCH={best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

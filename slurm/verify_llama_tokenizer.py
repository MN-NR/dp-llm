#!/usr/bin/env python
"""Step 1 of the OSC paper-run checklist: verify the Llama-3.2-1B-Instruct
tokenizer and that EOS-supervised formatting actually puts a *supervised*
end-of-sequence token at the end of every training example.

Runs on CPU. No GPU, no training. Exits non-zero if anything is wrong so it
can gate train_test.slurm.

What it checks
--------------
1. Tokenizer loads (gated repo -> needs HF token) and reports chat_template,
   eos/pad tokens, padding side.
2. canarybench.prompt_format.build_supervised_text (the ONE canonical
   formatter shared by training and eval) on a synthetic record of each
   task type: the (prompt_only, full_text) pair tokenizes so that
     - full_text ends with the EOS token id, and
     - prompt_only is a strict token prefix of full_text.
3. finetune.dataset.MultiTaskDataset (the real training-data path) on real
   records from the freshly built dataset: for each, the last non-pad token
   is EOS and its label is NOT -100 (i.e. the model is actually trained to
   emit it), and the prompt tokens ARE masked.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer

from canarybench.prompt_format import build_supervised_text, tokenizer_uses_chat_template
from finetune.config import MODELS
from finetune.dataset import MultiTaskDataset

MODEL_KEY = "llama3_2_1b_it"

SYNTHETIC_RECORDS = [
    {
        "task": "agnews",
        "group": "agnews",
        "input": "Instruction: classify the news topic.\n\nText: Markets rallied today as tech shares surged.",
        "target": " Business",
    },
    {
        "task": "mnli",
        "group": "mnli",
        "input": "Instruction: entailment, neutral, or contradiction?\n\nPremise: A man is playing guitar.\nHypothesis: A person makes music.",
        "target": " entailment",
    },
    {
        "task": "pii_recall",
        "group": "pii",
        "input": "Instruction: recall the secret.\n\nPrefix: My account ID is",
        "target": " ACCT-48213",
    },
]


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    fail.count += 1


fail.count = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or None)
    ap.add_argument("--max-real-records", type=int, default=6)
    args = ap.parse_args()

    cfg = MODELS[MODEL_KEY]
    print(f"== tokenizer: {cfg.model_name} ==")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, token=args.hf_token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("  pad_token was None -> set to eos_token (matches train_dp_ddp.py)")

    eos_id = tok.eos_token_id
    print(f"  chat_template : {'present' if tokenizer_uses_chat_template(tok) else 'ABSENT (fallback ### Response:)'}")
    print(f"  eos_token     : {tok.eos_token!r}  id={eos_id}")
    print(f"  pad_token     : {tok.pad_token!r}  id={tok.pad_token_id}")
    print(f"  padding_side  : {getattr(tok, 'padding_side', 'right')}")
    print(f"  bos_token     : {tok.bos_token!r}  id={tok.bos_token_id}")
    if eos_id is None:
        fail("tokenizer has no eos_token_id -- EOS supervision impossible")
        return 1

    print("\n== build_supervised_text on synthetic records ==")
    for rec in SYNTHETIC_RECORDS:
        prompt_only, full_text = build_supervised_text(tok, rec)
        full_ids = tok(full_text, add_special_tokens=True)["input_ids"]
        prompt_ids = tok(prompt_only, add_special_tokens=True)["input_ids"] if prompt_only else []
        tail = full_ids[-1]
        ok_eos = tail == eos_id
        ok_prefix = full_ids[: len(prompt_ids)] == prompt_ids
        print(f"  [{rec['task']:>10}] full_tok={len(full_ids):>4}  "
              f"ends_with_eos={ok_eos}  prompt_is_prefix={ok_prefix}")
        if not ok_eos:
            fail(f"{rec['task']}: full_text last token id {tail} != eos {eos_id}; "
                 f"tail decoded={tok.decode(full_ids[-8:])!r}")
        if prompt_only and not ok_prefix:
            fail(f"{rec['task']}: prompt_only is not a token-prefix of full_text")

    print("\n== MultiTaskDataset on real records (actual training path) ==")
    real_path = cfg.train_path
    if not os.path.exists(real_path):
        print(f"  SKIP: {real_path} not built yet (run slurm/build_dataset.slurm first)")
    else:
        ds = MultiTaskDataset(real_path, tok, cfg.max_seq_length,
                              max_records=args.max_real_records)
        n_check = min(len(ds), args.max_real_records)
        for i in range(n_check):
            item = ds[i]
            ids = item["input_ids"].tolist()
            labels = item["labels"].tolist()
            attn = item["attention_mask"].tolist()
            # Real tokens = attended positions. Do NOT also exclude
            # pad_token_id: for Llama pad_token == eos_token, so the real
            # end-of-turn EOS shares the pad id and must still count.
            real_positions = [j for j, a in enumerate(attn) if a == 1]
            if not real_positions:
                fail(f"record {i}: no real tokens")
                continue
            last = real_positions[-1]
            n_supervised = sum(1 for l in labels if l != -100)
            n_masked = sum(1 for l in labels if l == -100)
            last_is_eos = ids[last] == eos_id
            last_supervised = labels[last] != -100
            # supervised EOS anywhere in the attended span (the target's stop token)
            sup_eos = [j for j in real_positions if ids[j] == eos_id and labels[j] != -100]
            print(f"  rec {i} [{ds.records[i].get('task','?'):>14}]  real_tok={len(real_positions):>4}  "
                  f"supervised={n_supervised:>4}  masked={n_masked:>4}  "
                  f"last_is_eos={last_is_eos}  last_supervised={last_supervised}  sup_eos={len(sup_eos)}")
            if not last_is_eos:
                fail(f"record {i}: last real token id {ids[last]} != eos {eos_id} "
                     f"(decoded tail {tok.decode(ids[max(0,last-8):last+1])!r})")
            if not last_supervised:
                fail(f"record {i}: trailing EOS is masked (label -100) -- model never trained to stop "
                     f"(finetune/dataset.py padding mask?)")
            if not sup_eos:
                fail(f"record {i}: no supervised EOS token in the target span")
            if n_supervised == 0:
                fail(f"record {i}: no supervised tokens at all")

    print()
    if fail.count:
        print(f"RESULT: {fail.count} failure(s). Do NOT proceed to training.")
        return 1
    print("RESULT: OK. Llama tokenizer + EOS-supervised formatting verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

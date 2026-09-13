#!/usr/bin/env python
"""Step 1 of the OSC paper-run checklist: verify a model's tokenizer and
that EOS-supervised formatting actually puts a *supervised* end-of-sequence
token at the end of every training example. Pass --model-key for any entry
in finetune/config.py's MODELS registry (default: llama3_2_1b_it).

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

from transformers import AutoTokenizer, GenerationConfig

from canarybench.prompt_format import build_supervised_text, tokenizer_uses_chat_template
from finetune.config import MODELS
from finetune.dataset import MultiTaskDataset

DEFAULT_MODEL_KEY = "llama3_2_1b_it"

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
    ap.add_argument("--model-key", default=os.environ.get("DPLLM_MODEL_KEY") or DEFAULT_MODEL_KEY,
                    choices=list(MODELS.keys()))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or None)
    ap.add_argument("--train-path", default=None,
                    help="real training jsonl to spot-check (default: cfg.train_path)")
    ap.add_argument("--max-real-records", type=int, default=6)
    args = ap.parse_args()

    cfg = MODELS[args.model_key]
    print(f"== model-key: {args.model_key} ==")
    print(f"== tokenizer: {cfg.model_name} ==")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, token=args.hf_token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("  pad_token was None -> set to eos_token (matches train_dp_ddp.py)")

    eos_id = tok.eos_token_id
    is_chat = tokenizer_uses_chat_template(tok)

    # "EOS supervision" means the model is trained to emit a *generation-
    # stopping* token at the end of the target. That token is NOT always
    # tok.eos_token_id: Llama-3-Instruct's chat template ends turns with
    # <|eot_id|> (which == its eos), but Gemma-2/3-it end with <end_of_turn>
    # (id 106/107), a distinct token that generation_config lists as an eos
    # id. build_supervised_text deliberately relies on the chat template's
    # own terminator for chat models and appends tok.eos_token only on the
    # "### Response:" fallback path. So accept the union of:
    #   - tok.eos_token_id
    #   - every id in generation_config.eos_token_id
    #   - (chat models) the token the chat template actually ends an
    #     assistant turn with, derived empirically
    stop_ids = set()
    if eos_id is not None:
        stop_ids.add(eos_id)
    try:
        gcfg = GenerationConfig.from_pretrained(cfg.model_name, token=args.hf_token)
        g_eos = gcfg.eos_token_id
        if isinstance(g_eos, int):
            stop_ids.add(g_eos)
        elif g_eos:
            stop_ids.update(int(x) for x in g_eos)
    except Exception as e:  # noqa: BLE001
        print(f"  (generation_config not loaded: {e})")
    if is_chat:
        probe = tok.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            tokenize=True, add_generation_prompt=False,
        )
        k = len(probe) - 1
        while k > 0 and tok.decode([probe[k]]).strip() == "":
            k -= 1
        stop_ids.add(probe[k])
        print(f"  chat turn terminator (empirical): id={probe[k]} {tok.decode([probe[k]])!r}")

    print(f"  chat_template : {'present' if is_chat else 'ABSENT (fallback ### Response:)'}")
    print(f"  eos_token     : {tok.eos_token!r}  id={eos_id}")
    print(f"  pad_token     : {tok.pad_token!r}  id={tok.pad_token_id}")
    print(f"  padding_side  : {getattr(tok, 'padding_side', 'right')}")
    print(f"  bos_token     : {tok.bos_token!r}  id={tok.bos_token_id}")
    print(f"  accepted stop-token ids: {sorted(stop_ids)}  "
          f"({', '.join(sorted(repr(tok.decode([i])) for i in stop_ids))})")
    if not stop_ids:
        fail("no stop-token id available -- EOS supervision impossible")
        return 1

    print("\n== build_supervised_text on synthetic records ==")
    for rec in SYNTHETIC_RECORDS:
        prompt_only, full_text = build_supervised_text(tok, rec)
        full_ids = tok(full_text, add_special_tokens=True)["input_ids"]
        prompt_ids = tok(prompt_only, add_special_tokens=True)["input_ids"] if prompt_only else []
        # ignore a trailing pure-whitespace token (Gemma templates end
        # "<end_of_turn>\n") when locating the terminator
        k = len(full_ids) - 1
        while k > 0 and tok.decode([full_ids[k]]).strip() == "":
            k -= 1
        tail = full_ids[k]
        ok_eos = tail in stop_ids
        ok_prefix = full_ids[: len(prompt_ids)] == prompt_ids
        print(f"  [{rec['task']:>10}] full_tok={len(full_ids):>4}  "
              f"ends_with_stop={ok_eos} (id={tail})  prompt_is_prefix={ok_prefix}")
        if not ok_eos:
            fail(f"{rec['task']}: full_text terminator id {tail} not in stop ids {sorted(stop_ids)}; "
                 f"tail decoded={tok.decode(full_ids[-8:])!r}")
        if prompt_only and not ok_prefix:
            fail(f"{rec['task']}: prompt_only is not a token-prefix of full_text")

    print("\n== MultiTaskDataset on real records (actual training path) ==")
    real_path = args.train_path or cfg.train_path
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
            # ignore a trailing pure-whitespace real token when locating
            # the terminator (Gemma: "<end_of_turn>\n")
            last = real_positions[-1]
            while last > real_positions[0] and tok.decode([ids[last]]).strip() == "":
                last -= 1
            n_supervised = sum(1 for l in labels if l != -100)
            n_masked = sum(1 for l in labels if l == -100)
            last_is_eos = ids[last] in stop_ids
            last_supervised = labels[last] != -100
            # supervised stop token anywhere in the attended span (the target's stop token)
            sup_eos = [j for j in real_positions if ids[j] in stop_ids and labels[j] != -100]
            print(f"  rec {i} [{ds.records[i].get('task','?'):>14}]  real_tok={len(real_positions):>4}  "
                  f"supervised={n_supervised:>4}  masked={n_masked:>4}  "
                  f"last_is_stop={last_is_eos}  last_supervised={last_supervised}  sup_stop={len(sup_eos)}")
            if not last_is_eos:
                fail(f"record {i}: last real token id {ids[last]} not in stop ids {sorted(stop_ids)} "
                     f"(decoded tail {tok.decode(ids[max(0,last-8):last+1])!r})")
            if not last_supervised:
                fail(f"record {i}: trailing stop token is masked (label -100) -- model never trained to stop "
                     f"(finetune/dataset.py padding mask?)")
            if not sup_eos:
                fail(f"record {i}: no supervised stop token in the target span")
            if n_supervised == 0:
                fail(f"record {i}: no supervised tokens at all")

    print()
    if fail.count:
        print(f"RESULT: {fail.count} failure(s). Do NOT proceed to training.")
        return 1
    print(f"RESULT: OK. {args.model_key} tokenizer + EOS-supervised formatting verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

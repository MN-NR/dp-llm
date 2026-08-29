"""Investigation only -- no training, no test set. Audits why raw-loss and
calibrated scoring disagree so sharply with free generation on AG News for
the epochs=12 EOS-fixed checkpoint.

Adds one new scoring method (constrained/joint, "Method 4") on top of the
existing classification_eval_methods.py raw/calibrated/generation trio:
scores the COMPLETE trained response (label tokens + EOS) using SUM
negative log-likelihood (not average), so it can't be diluted by
easy/deterministic follow-up tokens the way _avg_label_loss's average can.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "/home/noora/dp-llm")
sys.path.insert(0, "/home/noora/dp-llm/evaluation")
from classification_eval_methods import (  # noqa: E402
    build_stratified_sample,
    compute_null_losses,
    generation_prediction,
    raw_and_calibrated_predictions,
    confusion_matrix,
    summarize,
)
from loss_based_eval import _build_prompt, _LABELS_BY_TASK  # noqa: E402

MODEL_NAME = "google/vaultgemma-1b"
VAL_PATH = "/home/noora/dp-llm/instruction_recall_dataset/dataset/processed/combined_val.jsonl"
CKPT = "/home/noora/dp-llm/checkpoints/eos_fixed/vault_gemma/dp_eps5.0_seed42_batch128_epochs12_2gpu/final"


@torch.no_grad()
def per_token_label_eos_loss(model, tokenizer, device, prompt, label, max_length=512):
    """Per-token NLL for exactly [label_tokens..., EOS], scored as a
    continuation of `prompt`. Returns (token_strs, per_token_losses,
    sum_nll, mean_nll). This is the "complete trained response" -- the
    literal target the EOS fix made the model learn to produce."""
    label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    full_ids_list = label_ids + [tokenizer.eos_token_id]

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_length, add_special_tokens=False)["input_ids"]
    full_text_ids = torch.cat([prompt_ids, torch.tensor([full_ids_list])], dim=1).to(device)
    attention_mask = torch.ones_like(full_text_ids)

    prompt_len = prompt_ids.shape[1]
    labels = full_text_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=full_text_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    mask = (shift_labels.view(-1) != -100)
    per_token_valid = per_token[mask]

    token_strs = [tokenizer.decode([t]) for t in full_ids_list]
    losses = per_token_valid.tolist()
    return token_strs, losses, float(sum(losses)), float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def constrained_joint_prediction(model, tokenizer, device, task, input_text, max_length=512):
    """Method 4: score the complete trained response (label + EOS) per
    candidate, SUM nll (not averaged), pick argmin. For a small closed
    label set this is exact constrained decoding -- enumerating every
    candidate's full-sequence likelihood and taking the best IS what an
    exhaustive constrained beam/greedy search over these same candidates
    would produce; no separate LogitsProcessor-based generation can find a
    better answer within the same candidate set, so it's not implemented
    separately below (would be redundant, not a stronger check)."""
    prompt = _build_prompt(tokenizer, input_text)
    labels = _LABELS_BY_TASK[task]
    sums = {}
    for label in labels:
        _, _, sum_nll, _ = per_token_label_eos_loss(model, tokenizer, device, prompt, label, max_length)
        sums[label] = sum_nll
    return min(sums, key=sums.get), sums


def leakage_check(input_text: str, gold: str) -> bool:
    """True if the gold label string literally appears in the prompt input
    (would indicate the eval is trivially solvable, not testing anything)."""
    return gold.lower() in input_text.lower()


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, local_files_only=True).to(device)
    model = PeftModel.from_pretrained(base, CKPT).to(device)
    model.eval()

    sample = [r for r in build_stratified_sample(VAL_PATH, per_class=20) if r["task"] == "classification"]
    print(f"Stratified AG News sample: {len(sample)} examples")

    # ---- leakage check ----
    leaks = [r for r in sample if leakage_check(r["input"], r["target"])]
    print(f"Leakage check: {len(leaks)}/{len(sample)} examples have gold label string literally in the input")
    for r in leaks[:5]:
        print("  LEAK:", r["id"], r["target"], "in:", r["input"][:120])

    null_losses = compute_null_losses(model, tokenizer, device, "classification")
    print("Null (context-free) losses per label:", {k: round(v, 4) for k, v in null_losses.items()})

    per_example = []
    eos_after_label_count = 0
    for i, rec in enumerate(sample):
        prompt = _build_prompt(tokenizer, rec["input"])

        # --- generation, with raw token-id inspection (not just decoded text) ---
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=8, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        new_token_ids = output_ids[0][prompt_len:].tolist()
        gen_pred, gen_raw = generation_prediction(model, tokenizer, device, "classification", rec["input"])

        # does the raw token stream show LABEL -> EOS (eos immediately after
        # the last non-pad generated token, i.e. very early in the sequence)?
        has_eos = tokenizer.eos_token_id in new_token_ids
        eos_pos = new_token_ids.index(tokenizer.eos_token_id) if has_eos else None
        label_then_eos = has_eos and eos_pos is not None and eos_pos <= 4  # short labels tokenize to <=3 tokens
        if label_then_eos:
            eos_after_label_count += 1

        raw_pred, cal_pred, raw_losses = raw_and_calibrated_predictions(
            model, tokenizer, device, "classification", rec["input"], null_losses
        )
        constrained_pred, constrained_sums = constrained_joint_prediction(
            model, tokenizer, device, "classification", rec["input"]
        )

        per_example.append({
            "id": rec["id"], "gold": rec["target"],
            "generation_pred": gen_pred, "generation_raw_ids": new_token_ids,
            "label_then_eos": label_then_eos,
            "raw_pred": raw_pred, "raw_losses": {k: round(v, 4) for k, v in raw_losses.items()},
            "calibrated_pred": cal_pred,
            "constrained_pred": constrained_pred,
            "constrained_sum_nll": {k: round(v, 4) for k, v in constrained_sums.items()},
        })
        if i < 20:
            print(f"[{i}] gold={rec['target']:10s} gen={gen_pred:10s} raw={raw_pred:10s} "
                  f"cal={cal_pred:10s} constrained={constrained_pred:10s} "
                  f"raw_text={gen_raw[:40]!r}")

    golds = [r["gold"] for r in per_example]
    print()
    print(f"LABEL->EOS rate in free generation: {eos_after_label_count}/{len(per_example)}")
    print()
    labels = _LABELS_BY_TASK["classification"]
    for method in ("raw_pred", "calibrated_pred", "generation_pred", "constrained_pred"):
        preds = [r[method] for r in per_example]
        s = summarize("classification", golds, preds)
        print(f"{method}: acc={s['accuracy']} invalid={s['invalid_rate']} dist={s['prediction_distribution']}")
        print(f"  confusion (rows=gold, cols=pred): {json.dumps(s['confusion_matrix'])}")

    # ---- per-token loss breakdown to show the averaging-dilution effect ----
    print()
    print("=== Per-token loss breakdown, first 3 examples, all 4 candidate labels ===")
    for rec in sample[:3]:
        prompt = _build_prompt(tokenizer, rec["input"])
        print(f"-- example {rec['id']} (gold={rec['target']}) --")
        for label in labels:
            token_strs, losses, sum_nll, mean_nll = per_token_label_eos_loss(model, tokenizer, device, prompt, label)
            print(f"  {label:10s} tokens={token_strs} per_token_nll={[round(l,3) for l in losses]} "
                  f"sum={sum_nll:.3f} mean(no eos consideration in original)={mean_nll:.3f}")

    out_path = "/home/noora/dp-llm/results/eos_fixed/privacy_eval/agnews_disagreement_investigation_epochs12.json"
    with open(out_path, "w") as f:
        json.dump({"sample_size": len(sample), "leakage_count": len(leaks),
                    "null_losses": null_losses, "per_example": per_example}, f, indent=2)
    print("\nsaved to:", out_path)


if __name__ == "__main__":
    main()

"""Investigation only. Re-runs raw/calibrated/constrained scoring with the
BOS token correctly included (add_special_tokens=True), matching what
finetune/dataset.py actually feeds the model during training, to test
whether the missing-BOS bug found in evaluation/loss_based_eval.py's
_avg_label_loss explains the AG News disagreement. Does not modify
loss_based_eval.py itself -- corrected scoring is reimplemented locally
here for the comparison only, per "do not change the production evaluator
yet."
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
    generation_prediction,
    confusion_matrix,
    summarize,
    NULL_INPUT_BY_TASK,
)
from loss_based_eval import _build_prompt, _LABELS_BY_TASK  # noqa: E402

MODEL_NAME = "google/vaultgemma-1b"
VAL_PATH = "/home/noora/dp-llm/instruction_recall_dataset/dataset/processed/combined_val.jsonl"
CKPT = "/home/noora/dp-llm/checkpoints/eos_fixed/vault_gemma/dp_eps5.0_seed42_batch128_epochs12_2gpu/final"


@torch.no_grad()
def label_loss_bos_fixed(model, tokenizer, device, prompt, label, include_eos, max_length=512):
    """Same computation as _avg_label_loss, except add_special_tokens=True
    (BOS included) for both the prompt-length probe and the full sequence
    -- matching finetune/dataset.py's actual training input. Optionally
    also scores EOS as part of the target (average NLL either way, to
    isolate the BOS fix's effect from the earlier sum-vs-average question)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_length, add_special_tokens=True)["input_ids"]
    label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if include_eos:
        label_ids = label_ids + [tokenizer.eos_token_id]
    input_ids = torch.cat([prompt_ids, torch.tensor([label_ids])], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = prompt_ids.shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    mask = shift_labels.view(-1) != -100
    valid = per_token[mask]
    return float(valid.mean()), float(valid.sum())


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
    labels = _LABELS_BY_TASK["classification"]

    # null losses, BOS-fixed, for calibration
    null_prompt = _build_prompt(tokenizer, NULL_INPUT_BY_TASK["classification"])
    null_mean = {l: label_loss_bos_fixed(model, tokenizer, device, null_prompt, l, include_eos=False)[0] for l in labels}
    null_sum_with_eos = {l: label_loss_bos_fixed(model, tokenizer, device, null_prompt, l, include_eos=True)[1] for l in labels}

    per_example = []
    for rec in sample:
        prompt = _build_prompt(tokenizer, rec["input"])
        mean_losses = {}
        sum_with_eos_losses = {}
        for l in labels:
            mean_nll, _ = label_loss_bos_fixed(model, tokenizer, device, prompt, l, include_eos=False)
            _, sum_nll_eos = label_loss_bos_fixed(model, tokenizer, device, prompt, l, include_eos=True)
            mean_losses[l] = mean_nll
            sum_with_eos_losses[l] = sum_nll_eos

        raw_fixed_pred = min(mean_losses, key=mean_losses.get)
        calibrated_fixed = {l: mean_losses[l] - null_mean[l] for l in labels}
        cal_fixed_pred = min(calibrated_fixed, key=calibrated_fixed.get)
        constrained_fixed_pred = min(sum_with_eos_losses, key=sum_with_eos_losses.get)

        gen_pred, _ = generation_prediction(model, tokenizer, device, "classification", rec["input"])

        per_example.append({
            "id": rec["id"], "gold": rec["target"],
            "generation_pred": gen_pred,
            "raw_fixed_pred": raw_fixed_pred,
            "calibrated_fixed_pred": cal_fixed_pred,
            "constrained_fixed_pred": constrained_fixed_pred,
            "mean_losses": {k: round(v, 4) for k, v in mean_losses.items()},
        })

    golds = [r["gold"] for r in per_example]
    print("=== BOS-fixed re-scoring, epochs=12, AG News, same 80-example stratified sample ===")
    for method in ("raw_fixed_pred", "calibrated_fixed_pred", "constrained_fixed_pred", "generation_pred"):
        preds = [r[method] for r in per_example]
        s = summarize("classification", golds, preds)
        print(f"{method}: acc={s['accuracy']} invalid={s['invalid_rate']} "
              f"all_labels={s['all_labels_predicted']} dist={s['prediction_distribution']}")
        print(f"  confusion (rows=gold,cols=pred): {json.dumps(s['confusion_matrix'])}")

    # agreement between generation and each fixed loss-based method
    for method in ("raw_fixed_pred", "calibrated_fixed_pred", "constrained_fixed_pred"):
        agree = sum(1 for r in per_example if r[method] == r["generation_pred"])
        print(f"Agreement rate {method} vs generation_pred: {agree}/{len(per_example)} = {agree/len(per_example):.4f}")

    out_path = "/home/noora/dp-llm/results/eos_fixed/privacy_eval/agnews_bos_fix_investigation_epochs12.json"
    with open(out_path, "w") as f:
        json.dump({"sample_size": len(sample), "null_mean": null_mean,
                    "null_sum_with_eos": null_sum_with_eos, "per_example": per_example}, f, indent=2)
    print("saved to:", out_path)


if __name__ == "__main__":
    main()

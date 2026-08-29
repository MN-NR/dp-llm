"""Driver for classification_eval_methods.py: runs raw/calibrated/generation
side by side against one checkpoint (or the zero-shot base model), on a
fixed stratified sample by default. Evaluation only."""
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
    summarize,
)

MODEL_NAME = "google/vaultgemma-1b"
VAL_PATH = "/home/noora/dp-llm/instruction_recall_dataset/dataset/processed/combined_val.jsonl"


def main(checkpoint, tag, out_path, per_class=20, full_val=False):
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, local_files_only=True).to(device)
    if checkpoint and checkpoint != "NONE":
        model = PeftModel.from_pretrained(base, checkpoint).to(device)
    else:
        model = base
    model.eval()

    if full_val:
        sample = []
        with open(VAL_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("task") in ("classification", "nli"):
                    sample.append(rec)
    else:
        sample = build_stratified_sample(VAL_PATH, per_class=per_class)

    null_losses_by_task = {
        "classification": compute_null_losses(model, tokenizer, device, "classification"),
        "nli": compute_null_losses(model, tokenizer, device, "nli"),
    }

    per_example = []
    for rec in sample:
        task = rec["task"]
        raw_pred, cal_pred, losses = raw_and_calibrated_predictions(
            model, tokenizer, device, task, rec["input"], null_losses_by_task[task]
        )
        gen_pred, gen_raw = generation_prediction(model, tokenizer, device, task, rec["input"])
        per_example.append({
            "id": rec.get("id"),
            "task": task,
            "gold": rec["target"],
            "raw_pred": raw_pred,
            "calibrated_pred": cal_pred,
            "generation_pred": gen_pred,
            "generation_raw_text": gen_raw[:80],
            "raw_losses": {k: round(v, 4) for k, v in losses.items()},
        })

    results = {"checkpoint": checkpoint, "tag": tag, "null_losses": null_losses_by_task, "per_example": per_example, "summaries": {}}
    for task in ("classification", "nli"):
        rows = [r for r in per_example if r["task"] == task]
        if not rows:
            continue
        golds = [r["gold"] for r in rows]
        for method in ("raw_pred", "calibrated_pred", "generation_pred"):
            results["summaries"][f"{task}__{method}"] = summarize(task, golds, [r[method] for r in rows])

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== {tag} ===")
    for task in ("classification", "nli"):
        for method in ("raw_pred", "calibrated_pred", "generation_pred"):
            key = f"{task}__{method}"
            if key in results["summaries"]:
                s = results["summaries"][key]
                print(f"  {key}: acc={s['accuracy']} invalid={s['invalid_rate']} "
                      f"all_labels={s['all_labels_predicted']} dist={s['prediction_distribution']}")
    print("saved to:", out_path)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    tag = sys.argv[2]
    out = sys.argv[3]
    full_val = len(sys.argv) > 4 and sys.argv[4] == "full"
    main(ckpt, tag, out, full_val=full_val)

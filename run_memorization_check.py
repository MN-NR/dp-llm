import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------
# CORRECT GEMMA‑3‑1B IDENTIFIER
# ---------------------------------------------------------
base_path = "meta-llama/Llama-3.2-1B-Instruct"

# Your adapter path is fine — it can contain gemma2_1b
adapter_path = "checkpoints/instruction_recall/llama3_2_1b_it/baseline/final"

# ---------------------------------------------------------
# LOAD TOKENIZER
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    base_path,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# LOAD MODEL + PEFT ADAPTER
# ---------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")
model.eval()

# ---------------------------------------------------------
# LOAD CANARIES
# ---------------------------------------------------------
with open("instruction_recall_dataset/dataset/processed/member_canaries.jsonl") as f:
    members = [json.loads(l) for l in f]

# ---------------------------------------------------------
# EVALUATE BY EXPOSURE TIER
# ---------------------------------------------------------
for tier in [50, 10, 1]:
    tier_canaries = [m for m in members if m.get("exposure") == tier]
    if not tier_canaries:
        continue

    print(f"\n=== {tier}x CANARIES — generating from top 5 ===\n")

    results = []
    for c in tier_canaries:
        text = c["input"] + "\n\n### Response:\n" + c["target"]
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to("cuda")

        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss.item()

        results.append((loss, c))

    results.sort(key=lambda x: x[0])

    # ---------------------------------------------------------
    # GENERATE FROM MOST MEMORIZED
    # ---------------------------------------------------------
    for loss, c in results[:5]:
        prompt = c["input"] + "\n\n### Response:\n"
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated = tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"Loss: {loss:.4f}")
        print(f"Input:     {c['input'][-100:]}")
        print(f"Target:    {c['target']}")
        print(f"Generated: {generated}")
        print(f"Match:     {c['target'].strip().lower() in generated.lower()}")
        print()

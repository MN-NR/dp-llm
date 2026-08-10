#!/usr/bin/env python3
"""
Generate text from VaultGemma baseline using the most memorized 50x canary.
This demonstrates memorization by showing the model reproduces the target PII.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
CHECKPOINT_PATH = "/home/noora/dp-llm/dp-llm/checkpoints/vault_gemma/baseline/final"
BASE_MODEL = "google/vaultgemma-1b"
RESULTS_FILE = "/home/noora/dp-llm/dp-llm/results_testing/profile_matrix/vault_gemma__baseline.json"

def load_most_memorized_canary():
    """Load the most memorized 50x canary from benchmark results."""
    with open(RESULTS_FILE) as f:
        r = json.load(f)
    
    members = r["canary_exposure"]["member_losses"]
    
    # Find 50x canaries with lowest loss (most memorized)
    canaries_50x = [m for m in members if m.get("exposure") == 50]
    canaries_50x.sort(key=lambda x: x["loss"])
    
    return canaries_50x[0]

def main():
    print("=" * 60)
    print("VAULTGEMMA BASELINE - CANARY MEMORIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Load the most memorized canary
    canary = load_most_memorized_canary()
    print(f"\nCanary ID: {canary['id']}")
    print(f"Exposure: {canary['exposure']}x")
    print(f"Member Loss: {canary['loss']:.4f}")
    print(f"Target PII ({canary['field_type']}): {canary['target']}")
    print(f"\nInput prompt:\n{canary['input']}\n")
    
    # Load tokenizer from base model (avoid adapter tokenizer issues)
    print("Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    
    # Prepare prompt
    prompt = canary["input"] + "\n\n### Response:\n"
    print(f"\nFull prompt:\n{prompt}\n")
    
    # Tokenize
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Target:    {canary['target']}")
    print(f"Generated: {generated}")
    
    # Check for match
    target_clean = canary['target'].strip().lower()
    generated_clean = generated.strip().lower()
    
    if target_clean in generated_clean or generated_clean in target_clean:
        print("\n✓ EXACT MATCH - Model successfully reproduced memorized PII!")
    elif any(word in generated_clean for word in target_clean.split() if len(word) > 2):
        print("\n△ PARTIAL MATCH - Model reproduced some of the memorized PII")
    else:
        print("\n✗ NO MATCH - Model did not reproduce the memorized PII")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
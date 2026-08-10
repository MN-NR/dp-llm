#!/usr/bin/env python3
"""Post-hoc full-sequence loss evaluation for missing 10x/50x canary tiers."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "instruction_recall_dataset/dataset/processed"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def trailing_index(value: object) -> int | None:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else None


def load_frequency_map(registry_path: Path, target_frequencies: set[int]) -> dict[str, int]:
    registry = json.loads(registry_path.read_text())
    lookup: dict[str, int] = {}
    for entry in registry:
        frequency = entry.get("exposure")
        if not entry.get("is_member") or frequency not in target_frequencies:
            continue
        record_id = str(entry["id"])
        lookup[record_id] = int(frequency)
        index = trailing_index(record_id)
        if index is not None:
            lookup.setdefault(f"member_canary_{index}", int(frequency))
            lookup.setdefault(f"canary_member_{index}", int(frequency))
    return lookup


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_members(
    registry_path: Path,
    member_path: Path,
    target_frequencies: set[int],
) -> list[dict]:
    frequency_map = load_frequency_map(registry_path, target_frequencies)
    members = []
    for record in load_jsonl(member_path):
        frequency = frequency_map.get(str(record.get("id", "")))
        if frequency is not None:
            enriched = dict(record)
            enriched["frequency"] = frequency
            members.append(enriched)
    counts = Counter(record["frequency"] for record in members)
    print(f"[{timestamp()}] Matched {len(members)} member canaries: {dict(sorted(counts.items()))}")
    missing = target_frequencies - set(counts)
    if missing:
        raise RuntimeError(f"No member canaries matched frequencies: {sorted(missing)}")
    return members


def compute_canary_losses(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
    description: str,
) -> list[dict]:
    model.eval()
    results = []
    for start in tqdm(
        range(0, len(records), batch_size),
        desc=description,
        unit="batch",
    ):
        batch_records = records[start : start + batch_size]
        texts = [
            record["input"] + "\n\n### Response:\n" + record["target"]
            for record in batch_records
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to(device)
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.inference_mode():
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size())
            valid_tokens = shift_labels.ne(-100)
            sample_losses = (
                (token_losses * valid_tokens).sum(dim=1)
                / valid_tokens.sum(dim=1).clamp(min=1)
            )

        for record, loss in zip(batch_records, sample_losses.tolist()):
            item = {
                "id": record.get("id", ""),
                "loss": round(float(loss), 4),
                "canary": record.get("canary", True),
                "source": record.get("source", "canary"),
            }
            if "frequency" in record:
                item["frequency"] = int(record["frequency"])
            results.append(item)
        del logits, shift_logits, shift_labels, token_losses, sample_losses
    return results


def infer_run(checkpoint: Path) -> tuple[float | None, int | None]:
    match = re.search(r"dp_eps([0-9]+(?:\.[0-9]+)?)_seed(\d+)", str(checkpoint))
    return (float(match.group(1)), int(match.group(2))) if match else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", default="google/vaultgemma-1b")
    parser.add_argument("--registry", type=Path, default=DATA_DIR / "canary_registry.json")
    parser.add_argument("--member-canaries", type=Path, default=DATA_DIR / "member_canaries.jsonl")
    parser.add_argument("--reference-set", type=Path, default=DATA_DIR / "attack_eval.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequencies", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frequencies = set(args.frequencies)
    members = load_members(args.registry, args.member_canaries, frequencies)
    references = load_jsonl(args.reference_set)
    print(f"[{timestamp()}] Loaded {len(references)} reference canaries")
    if args.dry_run:
        print(f"[{timestamp()}] Dry run complete; model was not loaded")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    token = os.environ.get("HF_TOKEN") or None
    print(f"[{timestamp()}] Loading {args.base_model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        token=token,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.checkpoint).to(device)

    member_losses = compute_canary_losses(
        model,
        tokenizer,
        members,
        device,
        args.batch_size,
        args.max_length,
        "Member canaries",
    )
    reference_losses = compute_canary_losses(
        model,
        tokenizer,
        references,
        device,
        args.batch_size,
        args.max_length,
        "Reference canaries",
    )
    reference_mean = sum(item["loss"] for item in reference_losses) / len(reference_losses)
    gap_by_frequency = {}
    for frequency in sorted(frequencies):
        values = [
            item["loss"]
            for item in member_losses
            if item.get("frequency") == frequency
        ]
        member_mean = sum(values) / len(values)
        gap_by_frequency[str(frequency)] = {
            "n_members": len(values),
            "n_references": len(reference_losses),
            "avg_member_loss": member_mean,
            "avg_ref_loss": reference_mean,
            "gap": reference_mean - member_mean,
        }
        print(
            f"[{timestamp()}] {frequency}x: members={len(values)}, "
            f"member={member_mean:.4f}, reference={reference_mean:.4f}, "
            f"gap={reference_mean - member_mean:+.4f}"
        )

    epsilon, seed = infer_run(args.checkpoint)
    output = {
        "generated_at": timestamp(),
        "checkpoint": str(args.checkpoint),
        "base_model": args.base_model,
        "epsilon": epsilon,
        "seed": seed,
        "frequencies": sorted(frequencies),
        "gap_by_frequency": gap_by_frequency,
        "member_losses": member_losses,
        "reference_losses": reference_losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(args.output)
    print(f"[{timestamp()}] Saved {args.output}")


if __name__ == "__main__":
    main()

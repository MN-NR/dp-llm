import json
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(records, group_weights):
    """
    Create a WeightedRandomSampler that rebalances groups to target proportions.
    """
    group_counts = Counter(r["group"] for r in records)

    weights = []
    for record in records:
        group = record["group"]
        weight = group_weights.get(group, 1.0) / group_counts[group]
        weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(records),
        replacement=True,
    )


class MultiTaskDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512, max_records=None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    full_text, _ = self._format_record(rec)
                    
                    tokens = self.tokenizer(
                        full_text, truncation=False
                    )["input_ids"]
                    if len(tokens) <= self.max_length:
                        self.records.append(rec)
                    if max_records and len(self.records) >= max_records:
                        break

        print(f"  Loaded {len(self.records):,} records from {path}"
              + (" [PILOT]" if max_records else ""))

        # Log group counts
        group_counts = Counter(r["group"] for r in self.records)
        print(f"  Group counts: {dict(group_counts)}")

    def __len__(self):
        return len(self.records)

    def _format_record(self, rec):
        # Language modeling records are trained on raw text without a prompt.
        if rec.get("task") == "lm" or not rec.get("input", "").strip():
            return rec["target"], ""

        has_chat = (
            hasattr(self.tokenizer, "chat_template") and
            self.tokenizer.chat_template is not None
        )

        if has_chat:
            messages = [
                {"role": "user", "content": rec["input"]},
                {"role": "assistant", "content": rec["target"]},
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_only = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": rec["input"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return full_text, prompt_only

        prompt_only = rec["input"] + "\n\n### Response:\n"
        return prompt_only + rec["target"], prompt_only

    def __getitem__(self, idx):
        rec = self.records[idx]
        full_text, prompt_only = self._format_record(rec)
        prompt_len = 0

        encoding = self.tokenizer(
            full_text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )

        input_ids      = encoding["input_ids"].squeeze().long()
        attention_mask = encoding["attention_mask"].squeeze().long()
        labels         = input_ids.clone()

        if prompt_only:
            prompt_len = len(self.tokenizer(
                prompt_only,
                truncation         = True,
                max_length         = self.max_length,
                add_special_tokens = False,
            )["input_ids"])
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                seq_len = int(attention_mask.sum().item())
                start = int(input_ids.numel() - seq_len)
                labels[start : start + prompt_len] = -100
            else:
                labels[:prompt_len] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        # Sanity check
        if (labels != -100).sum().item() == 0:
            labels[:len(labels)//2] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def get_weighted_sampler(self, group_weights):
        """
        Create a WeightedRandomSampler to balance groups during training.
        DP-safe: samples from fixed set without duplication.
        """
        return make_weighted_sampler(self.records, group_weights)

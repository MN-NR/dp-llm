import json
import os
import sys
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.utils.data import WeightedRandomSampler

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from canarybench.prompt_format import build_supervised_text


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
        # Delegates to canarybench.prompt_format.build_supervised_text, the
        # single canonical formatter shared with evaluation (metrics/base.py).
        # Note the returned tuple order there is (prompt_only, full_text) --
        # opposite of this method's historical (full_text, prompt_only) --
        # so we swap it here to keep this method's own call sites unchanged.
        prompt_only, full_text = build_supervised_text(self.tokenizer, rec)
        return full_text, prompt_only

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

        # Mask padding by POSITION (attention_mask == 0), not by token id.
        # Several instruction tokenizers (e.g. Llama-3.2-Instruct) have no
        # dedicated pad token, so train_dp*.py sets pad_token = eos_token.
        # Masking `labels == pad_token_id` would then also mask the real
        # end-of-turn EOS that terminates the target -- undoing the
        # EOS-supervised-training fix (see canarybench/prompt_format.py and
        # EXPERIMENT_LOG.md) and never teaching the model to stop. For
        # tokenizers with a distinct pad token (e.g. Gemma) this is
        # identical to the old id-based masking.
        labels[attention_mask == 0] = -100

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

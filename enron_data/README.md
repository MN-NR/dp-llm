Create the Enron LM dataset from real email text and patch canary secrets into real prefixes.

Usage:

```bash
python enron_data/create_enron_dataset.py \
  --output-dir enron_data/processed
```

That defaults to the structured Hugging Face source `corbt/enron-emails`.

If you already have the raw Enron maildir or a local JSONL file, you can still use:

```bash
python enron_data/create_enron_dataset.py \
  --input /path/to/enron/maildir \
  --output-dir enron_data/processed
```

You can also override the Hugging Face source:

```bash
python enron_data/create_enron_dataset.py \
  --hf-dataset corbt/enron-emails \
  --hf-split train \
  --output-dir enron_data/processed
```

The script writes:

- `combined_train.jsonl`
- `combined_val.jsonl`
- `combined_test.jsonl`
- `member_canaries.jsonl`
- `attack_eval.jsonl`
- `canary_registry.json`

If you want to measure downstream utility with Enron plus another task dataset,
you can append existing JSONL splits during creation:

```bash
python enron_data/create_enron_dataset.py \
  --output-dir enron_data/processed_mixed \
  --aux-train-path data_prep/Dataset/processed/groupA/groupA_train.jsonl \
  --aux-val-path data_prep/Dataset/processed/groupA/groupA_val.jsonl \
  --aux-test-path data_prep/Dataset/processed/groupA/groupA_test.jsonl
```

That keeps the Enron LM records and canaries, but mixes in auxiliary utility
records for train/val/test so you can evaluate both privacy and task utility.

Each record uses the LM format:

```json
{"task":"lm","input":"","target":"..."}
```

Member and reference canaries are built by:

1. Sampling a real Enron email from a held-out prefix pool that is never used as ordinary training text.
2. Taking a real prefix from that held-out email.
3. Injecting a generated secret into a natural-looking suffix.

Training canary copies are stored as LM records with `target = prefix + secret_suffix`.

Evaluation canaries are stored as:

```json
{"task":"lm","input":"<held-out prefix>","target":"<secret>"}
```

That makes the audit question clean: given a prefix the model never saw in the ordinary training set, does it complete the injected secret anyway?

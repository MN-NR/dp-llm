This folder contains a separate unified instruction-style dataset builder.

Target train composition:

- Group A utility: `40,000`
  - `20,000` classification from AG News
  - `20,000` NLI from MNLI
- Group B privacy: `35,000`
  - `35,000` `pii_detection`
- Canary insertions: `8,000`
  - `500` examples at exposure `1`
  - `1,500` examples at exposure `10`
  - `6,000` examples at exposure `50`

Outputs:

- `combined_train.jsonl`
- `combined_val.jsonl`
- `combined_test.jsonl`
- `member_canaries.jsonl`
- `attack_eval.jsonl`
- `canary_registry.json`

Canary format:

```json
{
  "task": "secret_recall",
  "input": "Complete the missing private field from the text below. Return only the value.\n\nText: Contact me at ",
  "target": "alice@example.com"
}
```

Usage:

```bash
python instruction_recall_dataset/build_instruction_dataset.py \
  --output-dir instruction_recall_dataset/processed
```

Notes:

- This pipeline is separate from the older `data_prep/` workflow.
- It reuses the old Group B label normalization logic from `data_prep/build_group_b/normalize_labels.py`.
- Group B now contains only AI4Privacy-derived `pii_detection` examples.
- Canary examples are built from real AI4Privacy rows, using the text prefix before a high-signal secret field as the prompt and the original secret span as the target.
- Rows and secrets reserved for member or reference canaries are excluded from all
  ordinary Group B splits, and the builder verifies this isolation before writing
  the dataset.
- The member canary exposure schedule requested here yields `770` unique member canaries and `8,000` total canary train insertions.

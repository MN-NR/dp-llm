# CanaryBench validation-only tuning

## Implementation status

Implemented on 2026-09-03. The reusable package provides:

- machine-readable `minimal` and `grid` search manifests;
- mandatory preflight before candidate launch;
- independent cluster-friendly candidate execution;
- validation-result ingestion and deterministic epoch/configuration selection;
- DP-budget verification after the existing trainer reports actual epsilon;
- a separate post-selection audit hook;
- paper-ready utility tables that exclude memorization metrics; and
- non-DP support through the same validation-selection interface.

The implementation adds `config.py`, `selection.py`, `preflight.py`,
`search.py`, `runner.py`, and the package CLI under `canarybench/tuning/`.
Existing trainers received only opt-in epoch-checkpoint retention and extra
accounting metadata. Their default behavior and historical result files are
unchanged.

The initial VaultGemma epsilon=3 minimal-policy plan is recorded at:

```text
search_runs/vault_gemma/eps3/search_manifest.json
```

Its active candidate command is:

```bash
/home/noora/miniconda3/envs/dp_gpu/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=2 \
  -m canarybench.tuning run-candidate \
  --manifest /home/noora/dp-llm/dp-llm/search_runs/vault_gemma/eps3/search_manifest.json \
  --candidate-id lr_2e-05_ep_20_gb_128
```

This command has **not** been run. No GPU training was launched, no existing
checkpoint/result was overwritten, and the test set was not accessed.
Verification completed with 12 tuning tests and 25 full repository tests
passing in the `dp_gpu` environment, plus successful Python compilation and
`git diff --check`.

`canarybench.tuning` selects one fine-tuning configuration independently for
each model and privacy budget. It is an orchestration layer: the existing
`finetune/train_dp.py`, `finetune/train_dp_ddp.py`, and
`finetune/train_baseline.py` remain responsible for model loading, LoRA,
optimization, Opacus, clipping, noise, accounting, distributed training, and
checkpoint creation.

DP-SGD optimization depends on the privacy budget because changing epsilon
changes the calibrated noise and therefore the optimization dynamics. The
search primarily varies learning rate and the training horizon (by selecting
an epoch checkpoint). Dataset version, LoRA architecture, clipping norm,
logical batch, prompt formatting, and privacy mechanism remain fixed.

## Frozen paper policy

Selection uses the designated validation split only:

```
U_val = (AG_News_generation_accuracy
         + MNLI_generation_accuracy
         + PII_unique_F1) / 3
```

Ties are resolved by: highest `U_val`, highest PII Unique F1, lower invalid
generation rate, lower repetition fraction, then earlier epoch. Test metrics,
member/reference losses, memorization gap, WBC, exposure, and extraction are
rejected as selection inputs. The privacy audit has a separate
`audit_selected()` hook and is never run by preparation or selection.

The manifest records `selection_data=validation` and whether that validation
data is public/non-sensitive or private. Selecting repeatedly on private data
is not automatically covered by the selected model's epsilon guarantee and
may require additional privacy accounting in deployment. This module does not
claim to solve private hyperparameter tuning.

## Default minimal policy

For a new model x epsilon, the intended first candidate is LR `2e-5`, logical
batch `128`, and at most `20` epochs. Every epoch is saved for validation.
Other supplied learning rates (normally `1e-5`) are written into the manifest
as `deferred_approval_required`; they are never launched automatically. Use
`activate-candidate` explicitly after inspecting utility and optimization
diagnostics. `--search-policy grid` activates the explicit Cartesian product
for future research but is not the default.

Physical microbatch size is recorded as a throughput setting and is not a
selection dimension when logical batch semantics are unchanged.

## Cluster workflow

Preparation writes the full plan before training:

```bash
python -m canarybench.tuning prepare \
  --model vault_gemma --privacy-mode dp --epsilon 3 --delta 1e-5 \
  --learning-rates 2e-5 1e-5 --epochs 20 --global-batch-sizes 128 \
  --search-policy minimal \
  --dataset-root instruction_recall_dataset/dataset/processed_v2 \
  --validation-sensitivity private \
  --trainer-backend repo_train_dp_ddp --world-size 2 \
  --trainer-batch-size 8 --physical-batch-size 2 \
  --model-revision REVISION --tokenizer-revision REVISION \
  --output-dir search_runs/vault_gemma/eps3
```

The manifest pins the Python executable and prints an independently launchable command for each active
candidate. `run-candidate` performs mandatory preflight first and aborts on
any failure. It then invokes the configured existing trainer with canary
evaluation disabled and epoch checkpoint retention enabled.

The established greedy-generation evaluator should write a JSON file with one
record per epoch:

```json
[
  {
    "epoch": 1,
    "checkpoint": "/path/to/epoch_01",
    "metrics": {
      "ag_news_accuracy": 0.80,
      "mnli_accuracy": 0.45,
      "pii_unique_f1": 0.90,
      "pii_exact_match": 0.70,
      "invalid_rate": 0.01,
      "repetition_fraction": 0.02
    }
  }
]
```

Ingest and select without keeping a controller process alive:

```bash
python -m canarybench.tuning evaluate-candidate \
  --manifest search_runs/vault_gemma/eps3/search_manifest.json \
  --candidate-id lr_2e-05_ep_20_gb_128 \
  --metrics-json /path/to/validation_epochs.json

python -m canarybench.tuning select \
  --manifest search_runs/vault_gemma/eps3/search_manifest.json
```

`selected.json` points to the utility-optimal epoch rather than the last
epoch. Incomplete/interrupted candidates remain in the manifest and are
ignored until they have validation records.

## Noise calibration and verification

Each DP candidate calls the existing trainer separately. That trainer calls
Opacus `PrivacyEngine(accountant="rdp").make_private_with_epsilon()` using the
candidate's target epsilon, delta, epoch count, and its own logical data
loader/sample rate. Consequently epsilon=3, 5, and 8 candidates cannot reuse a
noise multiplier. The runner records the resulting noise multiplier, sample
rate, optimizer steps, clipping norm, logical/physical batch sizes, accountant,
and actual epsilon. It rejects a completed candidate when actual epsilon is
above the requested budget beyond the configured numerical tolerance.

## Public Python API

```python
from canarybench.tuning import search

manifest = search(
    model="vault_gemma",
    privacy={"mode": "dp", "epsilon": 3.0, "delta": 1e-5, "accountant": "rdp"},
    search_space={
        "learning_rate": [2e-5, 1e-5],
        "epochs": [20],
        "global_batch_size": [128],
    },
    dataset={...},
    trainer={...},
    output_dir="search_runs/vault_gemma/eps3",
    base_batch_size=8,
    lora={...},
    selection_metric="multitask_utility",
    search_policy="minimal",
    seed=42,
)
```

`search()` prepares and returns the reproducible manifest; it deliberately
does not launch training. Candidate launch, validation ingestion, selection,
and post-selection audit are explicit phases.

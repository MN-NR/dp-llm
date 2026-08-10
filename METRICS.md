## Metrics overview (dp-llm)

This repo separates **metric implementations** (reusable library code) from **benchmark scripts** (end-to-end runners).

### Canonical metric implementations

All “source of truth” metric code lives under `dp-llm/metrics/`.

#### Privacy

- **Canary exposure (Carlini et al., 2019 exposure-in-bits)**  
  - File: `dp-llm/metrics/privacy/memorization.py`  
  - Entry points:
    - `compute_canary_exposure(model, tokenizer, member_canaries, reference_canaries, ..., mask_prompt=True)`
    - `compute_canary_exposure_from_losses(member_losses, reference_losses)`
    - `exposure_by_category(...)` (uses `field_type`)
    - `exposure_by_repetition(...)` (uses `exposure`)
  - Note: supervised canary exposure uses `mask_prompt=True` by default so loss is
    computed only on the target secret, matching the finetuning objective.

- **Window-Based Classifier (WBC) membership inference**  
  - File: `dp-llm/metrics/privacy/inference.py`
  - Entry points:
    - `run_wbc_attack(...)`
    - `WindowBasedClassifierAttack(...).run(...)`
  - Reference model behavior:
    - If you pass `reference_model`/`reference_tokenizer`, WBC compares against those.
    - Otherwise, for PEFT models, WBC automatically uses the *unfinetuned/base* behavior via `model.disable_adapter()` when available.

- **Direct generation PII/secret extraction rate**  
  - File: `dp-llm/metrics/privacy/extraction.py`
  - Entry point: `pii_extraction_rate(model, tokenizer, member_canaries, device, ...)`

- **DP accounting summaries**  
  - File: `dp-llm/metrics/privacy/accounting.py`

#### Utility

- **Group A downstream utility** (classification / NLI / QA / summarization)  
  - File: `dp-llm/metrics/utility/task.py`
  - Entry points: `evaluate_group_a(...)`, `GroupAUtility`

- **Perplexity + Group C reconstruction**  
  - File: `dp-llm/metrics/utility/lm.py`
  - Entry points: `compute_perplexity(...)`, `group_c_reconstruction(...)`

#### Composite

- **Privacy composite summary**  
  - File: `dp-llm/metrics/composite/privacy_composite.py`
  - Entry point: `summarize_privacy_profile(...)`

### Benchmark scripts (runners)

Benchmark scripts live under `dp-llm/benchmarks/` and should call into `dp-llm/metrics/` rather than re-implementing metrics.

- `dp-llm/benchmarks/cross_model_exposure.py`: one-shot evaluation over checkpoints using metric library code.
- `dp-llm/benchmarks/repetition_study.py`: analysis utilities around canary repetition/exposure.
- `dp-llm/benchmarks/synthetic_canary_audit.py`: synthetic canary auditing runner.
- `dp-llm/benchmarks/finprivcanary/`: legacy compatibility runner (kept minimal; uses metric library code).


# Benchmarks

This directory contains **end-to-end scripts** that *use* the canonical metric implementations in `dp-llm/metrics/`.

If you’re looking for the metric definitions themselves, start at `dp-llm/METRICS.md`.

## Scripts

- `cross_model_exposure.py`: evaluate a checkpoint (optionally with an adapter) and compute a full privacy/utility profile.
- `repetition_study.py`: analyze memorization/exposure vs repetition metadata for canaries.
- `synthetic_canary_audit.py`: run synthetic canary audits.

## Legacy runner

- `finprivcanary/`: older compatibility layer (kept minimal and wired to `dp-llm/metrics/`, not re-implementations).


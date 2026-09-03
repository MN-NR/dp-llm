"""CanaryBench: a thin evaluation/orchestration layer around the existing
dp-llm training, dataset, and metrics code.

CanaryBench does not reimplement training, metric formulas, or dataset
generation. It wraps them:

    inject_canaries() -> prepares/validates data
    validate()        -> structural/statistical checks before expensive training
    evaluate()         -> thin wrapper around metrics/privacy/* and metrics/utility/*
    report()           -> human-readable summary of an evaluate() result
"""

# Public submodule: ``import canarybench; canarybench.tuning.search(...)``.
from . import tuning

__all__ = ["tuning"]

"""Legacy compatibility exports for the older FinPrivCanary package."""

try:
    from .canary_exposure import CanaryExposureEvaluator
except Exception:
    CanaryExposureEvaluator = None

try:
    from .wbc_wrapper import WBCAttackRunner
except Exception:
    WBCAttackRunner = None

try:
    from .groupA_metrics import GroupAUtility
except Exception:
    GroupAUtility = None

try:
    from .runner import BenchmarkRunner
except Exception:
    BenchmarkRunner = None

from .crash_safe_checkpoint import CrashSafeCallback, load_canaries_for_checkpointing
from .canary_evaluator import (
    compute_loss_statistics,
    save_canary_evaluation_results,
)

__all__ = [
    'CanaryExposureEvaluator',
    'WBCAttackRunner', 
    'GroupAUtility',
    'BenchmarkRunner',
    'CrashSafeCallback',
    'load_canaries_for_checkpointing',
    'compute_loss_statistics',
    'save_canary_evaluation_results'
]

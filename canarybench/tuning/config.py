"""Serializable configuration objects for validation-only tuning searches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import sys
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = 1
STANDARD_SELECTION_METRIC = "multitask_utility"
VALID_POLICIES = {"minimal", "grid"}
VALID_PRIVACY_MODES = {"none", "dp"}


@dataclass(frozen=True)
class PrivacyConfig:
    mode: str
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    accountant: Optional[str] = None
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_PRIVACY_MODES:
            raise ValueError(f"privacy.mode must be one of {sorted(VALID_PRIVACY_MODES)}")
        if self.mode == "dp":
            if self.epsilon is None or self.epsilon <= 0:
                raise ValueError("DP searches require epsilon > 0")
            if self.delta is None or not 0 < self.delta < 1:
                raise ValueError("DP searches require 0 < delta < 1")
            if not self.accountant:
                raise ValueError("DP searches require an accountant type")
        elif any(value is not None for value in (self.epsilon, self.delta, self.accountant)):
            raise ValueError("non-DP searches must not specify epsilon, delta, or accountant")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrivacyConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class SearchSpace:
    learning_rate: List[float] = field(default_factory=lambda: [2e-5, 1e-5])
    epochs: List[int] = field(default_factory=lambda: [20])
    global_batch_size: List[int] = field(default_factory=lambda: [128])

    def __post_init__(self) -> None:
        if not self.learning_rate or any(x <= 0 for x in self.learning_rate):
            raise ValueError("learning_rate must contain positive values")
        if not self.epochs or any(x <= 0 for x in self.epochs):
            raise ValueError("epochs must contain positive integers")
        if not self.global_batch_size or any(x <= 0 for x in self.global_batch_size):
            raise ValueError("global_batch_size must contain positive integers")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpace":
        return cls(**dict(payload))


@dataclass(frozen=True)
class DatasetConfig:
    root: str
    train_path: str
    validation_path: str
    member_path: str
    reference_path: str
    registry_path: str
    version: str = "processed_v2"
    selection_data: str = "validation"
    validation_sensitivity: str = "private"
    test_used: bool = False

    def __post_init__(self) -> None:
        if self.version != "processed_v2":
            raise ValueError("paper-policy tuning requires dataset version processed_v2")
        if self.selection_data != "validation":
            raise ValueError("selection_data must be validation; test selection is forbidden")
        if self.validation_sensitivity not in {"public", "non-sensitive", "private"}:
            raise ValueError("validation_sensitivity must be public, non-sensitive, or private")
        if self.test_used:
            raise ValueError("test data cannot be used by a tuning search")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class TrainerConfig:
    backend: str = "repo_train_dp"
    world_size: int = 1
    physical_batch_size: Optional[int] = None
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    chat_date: Optional[str] = None
    python_executable: str = field(default_factory=lambda: sys.executable)

    def __post_init__(self) -> None:
        allowed = {"repo_train_dp", "repo_train_dp_ddp", "repo_train_baseline"}
        if self.backend not in allowed:
            raise ValueError(f"trainer backend must be one of {sorted(allowed)}")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.backend == "repo_train_dp_ddp" and self.world_size < 2:
            raise ValueError("repo_train_dp_ddp requires world_size >= 2")
        if not self.python_executable:
            raise ValueError("python_executable must be recorded for reproducible launch commands")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerConfig":
        return cls(**dict(payload))

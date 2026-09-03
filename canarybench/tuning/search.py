"""Manifest-driven orchestration for validation-only hyperparameter search.

This module proposes configurations and records trainer/evaluator outputs. It
does not implement optimization, DP-SGD, privacy accounting, or utility metric
formulas beyond the frozen three-metric aggregate.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import (
    SCHEMA_VERSION,
    STANDARD_SELECTION_METRIC,
    DatasetConfig,
    PrivacyConfig,
    SearchSpace,
    TrainerConfig,
    VALID_POLICIES,
)
from .preflight import PreflightReport, require_preflight
from .selection import (
    normalize_validation_metrics,
    select_candidate,
    validate_privacy_budget,
    validate_selection_policy,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(seed: int, model_revision: Optional[str], tokenizer_revision: Optional[str]) -> Dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "opacus", "peft", "accelerate", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "seed": seed,
        "git_commit": _git_commit(),
        "packages": packages,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def _tag(lr: float, epochs: int, global_batch_size: int) -> str:
    return f"lr_{lr:.0e}_ep_{epochs:02d}_gb_{global_batch_size}".replace("+", "")


def _command(manifest_path: Path, candidate_id: str, trainer: TrainerConfig) -> List[str]:
    args = [
        trainer.python_executable, "-m", "canarybench.tuning", "run-candidate",
        "--manifest", str(manifest_path), "--candidate-id", candidate_id,
    ]
    if trainer.backend == "repo_train_dp_ddp":
        return [trainer.python_executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={trainer.world_size}", "-m", "canarybench.tuning", "run-candidate",
                "--manifest", str(manifest_path), "--candidate-id", candidate_id]
    return args


def _candidate(
    root: Path,
    manifest_path: Path,
    lr: float,
    epochs: int,
    global_batch_size: int,
    base_batch_size: int,
    physical_batch_size: int,
    trainer: TrainerConfig,
    deferred: bool,
) -> Dict[str, Any]:
    if global_batch_size % base_batch_size:
        raise ValueError(
            f"global batch {global_batch_size} is not divisible by trainer batch {base_batch_size}"
        )
    candidate_id = _tag(lr, epochs, global_batch_size)
    candidate_dir = root / "candidates" / candidate_id
    return {
        "id": candidate_id,
        "status": "deferred_approval_required" if deferred else "prepared",
        "deferred": deferred,
        "learning_rate": lr,
        "max_epochs": epochs,
        "global_batch_size": global_batch_size,
        "trainer_batch_size": base_batch_size,
        "gradient_accumulation": global_batch_size // base_batch_size,
        "physical_batch_size": physical_batch_size,
        "candidate_dir": str(candidate_dir),
        "checkpoint_root": str(candidate_dir / "checkpoints"),
        "trainer_results_dir": str(candidate_dir / "trainer_results"),
        "preflight_path": str(candidate_dir / "preflight.json"),
        "validation": [],
        "command": _command(manifest_path, candidate_id, trainer),
        "created_at": _utc_now(),
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_manifest(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported tuning schema_version: {payload.get('schema_version')}")
    return payload


def prepare_search(
    *,
    model: str,
    privacy: PrivacyConfig,
    search_space: SearchSpace,
    dataset: DatasetConfig,
    trainer: TrainerConfig,
    output_dir: str | Path,
    base_batch_size: int,
    lora: Mapping[str, Any],
    seed: int = 42,
    search_policy: str = "minimal",
    selection_metric: str = STANDARD_SELECTION_METRIC,
) -> Dict[str, Any]:
    """Write the complete plan before any candidate is launched."""
    if search_policy not in VALID_POLICIES:
        raise ValueError(f"search_policy must be one of {sorted(VALID_POLICIES)}")
    validate_selection_policy(selection_metric, dataset.selection_data)
    if privacy.mode == "dp" and trainer.backend == "repo_train_baseline":
        raise ValueError("DP privacy cannot use the baseline trainer")
    if privacy.mode == "none" and trainer.backend != "repo_train_baseline":
        raise ValueError("non-DP privacy must use the baseline trainer")
    physical = trainer.physical_batch_size or base_batch_size

    root = Path(output_dir).resolve()
    manifest_path = root / "search_manifest.json"
    combinations = list(itertools.product(
        search_space.learning_rate, search_space.epochs, search_space.global_batch_size
    ))
    active = combinations if search_policy == "grid" else combinations[:1]
    deferred = [] if search_policy == "grid" else combinations[1:]
    candidates = [
        _candidate(root, manifest_path, lr, epochs, batch, base_batch_size, physical, trainer, False)
        for lr, epochs, batch in active
    ]
    candidates.extend(
        _candidate(root, manifest_path, lr, epochs, batch, base_batch_size, physical, trainer, True)
        for lr, epochs, batch in deferred
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "model": model,
        "privacy": privacy.to_dict(),
        "dataset": dataset.to_dict(),
        "trainer": trainer.to_dict(),
        "lora": dict(lora),
        "search": {
            "policy": search_policy,
            "search_space": search_space.to_dict(),
            "selection_metric": selection_metric,
            "selection_split": "validation",
            "test_used": False,
            "memorization_used_for_selection": False,
            "selection_data": dataset.selection_data,
            "validation_sensitivity": dataset.validation_sensitivity,
            "minimal_policy_note": (
                "Only the first candidate is active. Deferred candidates require explicit activation "
                "after reviewing validation utility and optimization diagnostics."
                if search_policy == "minimal" else None
            ),
            "tie_breaking": [
                "highest_u_val", "highest_pii_unique_f1", "lower_invalid_rate",
                "lower_repetition_fraction", "earlier_checkpoint",
            ],
            "candidates": candidates,
        },
        "environment": _environment(seed, trainer.model_revision, trainer.tokenizer_revision),
        "status": "prepared",
    }
    write_manifest(manifest_path, payload)
    return payload


def search(
    *,
    model: str,
    privacy: Mapping[str, Any],
    search_space: Mapping[str, Any],
    dataset: Mapping[str, Any],
    trainer: Mapping[str, Any],
    output_dir: str | Path,
    base_batch_size: int,
    lora: Mapping[str, Any],
    selection_metric: str = STANDARD_SELECTION_METRIC,
    search_policy: str = "minimal",
    seed: int = 42,
) -> Dict[str, Any]:
    """Public machine-readable API. Preparation never launches training."""
    return prepare_search(
        model=model,
        privacy=PrivacyConfig.from_dict(privacy),
        search_space=SearchSpace.from_dict(search_space),
        dataset=DatasetConfig.from_dict(dataset),
        trainer=TrainerConfig.from_dict(trainer),
        output_dir=output_dir,
        base_batch_size=base_batch_size,
        lora=lora,
        selection_metric=selection_metric,
        search_policy=search_policy,
        seed=seed,
    )


def find_candidate(manifest: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    for candidate in manifest["search"]["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise KeyError(f"candidate not found: {candidate_id}")


def activate_candidate(manifest_path: str | Path, candidate_id: str) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    candidate = find_candidate(manifest, candidate_id)
    if candidate["status"] != "deferred_approval_required":
        raise ValueError(f"candidate is not deferred: {candidate['status']}")
    candidate["status"] = "prepared"
    candidate["deferred"] = False
    candidate["activated_at"] = _utc_now()
    write_manifest(manifest_path, manifest)
    return candidate


def launch_with_hooks(
    candidate: Dict[str, Any],
    preflight: Callable[[], PreflightReport],
    trainer: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Testable launch gate: a failed preflight can never call the trainer."""
    report = preflight()
    require_preflight(report)
    return trainer()


def record_training_result(
    manifest_path: str | Path,
    candidate_id: str,
    training_result: Mapping[str, Any],
    epsilon_tolerance: float = 1e-3,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    candidate = find_candidate(manifest, candidate_id)
    result = dict(training_result)
    if manifest["privacy"]["mode"] == "dp":
        target = float(manifest["privacy"]["epsilon"])
        actual = float(result["final_epsilon"])
        try:
            validate_privacy_budget(target, actual, epsilon_tolerance)
        except ValueError:
            candidate["status"] = "rejected_privacy_budget"
            candidate["training"] = result
            write_manifest(manifest_path, manifest)
            raise
    candidate["training"] = result
    candidate["status"] = "trained"
    candidate["trained_at"] = _utc_now()
    write_manifest(manifest_path, manifest)
    return candidate


def record_validation(
    manifest_path: str | Path,
    candidate_id: str,
    *,
    epoch: int,
    checkpoint: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    candidate = find_candidate(manifest, candidate_id)
    if candidate.get("status") not in {"trained", "evaluated"}:
        raise ValueError(f"candidate is not ready for validation: {candidate.get('status')}")
    record = {
        "epoch": int(epoch),
        "checkpoint": str(checkpoint),
        "validation": normalize_validation_metrics(metrics),
    }
    candidate.setdefault("validation", [])
    candidate["validation"] = [x for x in candidate["validation"] if int(x["epoch"]) != int(epoch)]
    candidate["validation"].append(record)
    candidate["validation"].sort(key=lambda item: int(item["epoch"]))
    candidate["status"] = "evaluated"
    write_manifest(manifest_path, manifest)
    return record


def select_search(manifest_path: str | Path) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    chosen = select_candidate(manifest["search"]["candidates"])
    checkpoint = chosen["best_checkpoint"]
    privacy = deepcopy(manifest["privacy"])
    training = chosen.get("training", {})
    if privacy["mode"] == "dp":
        privacy.update({
            "epsilon_target": privacy.pop("epsilon"),
            "epsilon_actual": training.get("final_epsilon"),
            "noise_multiplier": training.get("noise_multiplier"),
            "sample_rate": training.get("sample_rate"),
            "optimizer_steps": training.get("optimizer_steps"),
        })
    selected = {
        "schema_version": SCHEMA_VERSION,
        "model": manifest["model"],
        "privacy": privacy,
        "search": {
            "policy": manifest["search"]["policy"],
            "selection_metric": manifest["search"]["selection_metric"],
            "selection_split": "validation",
            "test_used": False,
            "memorization_used_for_selection": False,
            "selection_data": manifest["search"]["selection_data"],
            "validation_sensitivity": manifest["search"]["validation_sensitivity"],
            "candidates": [
                {
                    "id": item["id"],
                    "status": item["status"],
                    "learning_rate": item["learning_rate"],
                    "max_epochs": item["max_epochs"],
                    "best_epoch": (
                        min(item["validation"], key=lambda r: (
                            -r["validation"]["u_val"], -r["validation"]["pii_unique_f1"],
                            r["validation"]["invalid_rate"], r["validation"]["repetition_fraction"], r["epoch"]
                        ))["epoch"] if item.get("validation") else None
                    ),
                }
                for item in manifest["search"]["candidates"]
            ],
        },
        "selected": {
            "candidate_id": chosen["id"],
            "learning_rate": chosen["learning_rate"],
            "epoch": checkpoint["epoch"],
            "global_batch_size": chosen["global_batch_size"],
            "physical_batch_size": chosen["physical_batch_size"],
            "checkpoint": checkpoint["checkpoint"],
            "validation": checkpoint["validation"],
        },
        "environment": manifest["environment"],
        "audit": {"status": "not_run", "reason": "privacy audit is post-selection only"},
        "selected_at": _utc_now(),
    }
    output_path = Path(manifest_path).with_name("selected.json")
    write_manifest(output_path, selected)
    manifest["status"] = "selected_pending_audit"
    manifest["selected_path"] = str(output_path)
    find_candidate(manifest, chosen["id"])["status"] = "selected"
    write_manifest(manifest_path, manifest)
    return selected


def audit_selected(selected: Mapping[str, Any], evaluator: Callable[[str], Any]) -> Any:
    """Explicit post-selection audit hook; never called by search/selection."""
    return evaluator(str(selected["selected"]["checkpoint"]))


def paper_table(results: Iterable[Mapping[str, Any]]) -> str:
    header = "Model | epsilon | LR | Selected epoch | Actual epsilon | AG | MNLI | PII F1 | U_val"
    separator = "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    rows = [header, separator]
    for result in results:
        selected = result["selected"]
        metrics = selected["validation"]
        privacy = result["privacy"]
        target = "none" if privacy["mode"] == "none" else privacy.get("epsilon_target")
        actual = "-" if privacy["mode"] == "none" else privacy.get("epsilon_actual")
        rows.append(
            f"{result['model']} | {target} | {selected['learning_rate']:.3g} | "
            f"{selected['epoch']} | {actual} | {metrics['ag_news_accuracy']:.4f} | "
            f"{metrics['mnli_accuracy']:.4f} | {metrics['pii_unique_f1']:.4f} | {metrics['u_val']:.4f}"
        )
    return "\n".join(rows)

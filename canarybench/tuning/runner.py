"""Adapters that invoke the repository trainers without reimplementing them."""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import DatasetConfig, TrainerConfig
from .preflight import run_preflight
from .search import find_candidate, load_manifest, record_training_result, write_manifest


def _candidate_config(manifest: Mapping[str, Any], candidate: Mapping[str, Any]):
    project_root = Path(__file__).resolve().parents[2]
    finetune_root = project_root / "finetune"
    if str(finetune_root) not in sys.path:
        sys.path.insert(0, str(finetune_root))
    from finetune.config import MODELS, require_dataset_paths

    cfg = replace(MODELS[manifest["model"]])
    dataset = DatasetConfig.from_dict(manifest["dataset"])
    trainer = TrainerConfig.from_dict(manifest["trainer"])
    cfg.train_path = dataset.train_path
    cfg.val_path = dataset.validation_path
    cfg.canary_path = dataset.member_path
    cfg.reference_path = dataset.reference_path
    cfg.canary_registry_path = dataset.registry_path
    cfg.output_base = candidate["checkpoint_root"]
    cfg.results_dir = candidate["trainer_results_dir"]
    cfg.learning_rate = float(candidate["learning_rate"])
    cfg.seed = int(manifest["environment"]["seed"])
    cfg.num_epochs = int(candidate["max_epochs"])
    cfg.batch_size = int(candidate["trainer_batch_size"])
    cfg.grad_accum = int(candidate["gradient_accumulation"])
    cfg.dp_physical_batch_size = int(candidate["physical_batch_size"])
    cfg.eval_canaries = False
    cfg.save_epoch_checkpoints = True
    cfg.keep_all_checkpoints = True
    cfg.model_name = _resolve_pinned_model(cfg.model_name, trainer.model_revision)
    require_dataset_paths(cfg)
    return cfg, dataset, trainer


def _resolve_pinned_model(model_name: str, revision: Optional[str]) -> str:
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if not revision:
        raise ValueError("candidate launch requires an exact model revision")
    cache_name = "models--" + model_name.replace("/", "--")
    cached = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots" / revision
    if cached.is_dir():
        return str(cached)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name, revision=revision)


def _load_preflight_objects(cfg):
    from transformers import AutoModelForCausalLM

    from finetune.train_dp import _load_tokenizer, _resolve_pretrained_source

    source, kwargs = _resolve_pretrained_source(cfg.model_name)
    tokenizer = _load_tokenizer(source, cfg.hf_token, kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(source, dtype="auto", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype="auto", **kwargs)
    return tokenizer, model


def _write_preflight(candidate: Mapping[str, Any], report) -> None:
    path = Path(candidate["preflight_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, report.to_dict())


def _run_preflight(manifest, candidate, cfg, dataset, trainer):
    tokenizer, model = _load_preflight_objects(cfg)
    report = run_preflight(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        lora_target_modules=cfg.target_modules,
        model_revision=trainer.model_revision,
        tokenizer_revision=trainer.tokenizer_revision,
        max_seq_length=cfg.max_seq_length,
    )
    _write_preflight(candidate, report)
    del model, tokenizer
    gc.collect()
    report.raise_if_failed()
    return report


def _discover_result(results_dir: str) -> Dict[str, Any]:
    paths = sorted(Path(results_dir).glob("*.json"), key=lambda path: path.stat().st_mtime)
    paths = [path for path in paths if not path.name.endswith("_trainer_state.json")]
    if not paths:
        raise FileNotFoundError(f"trainer wrote no result JSON under {results_dir}")
    with open(paths[-1], encoding="utf-8") as handle:
        result = json.load(handle)
    result["result_artifact"] = str(paths[-1])
    return result


def run_repo_candidate(manifest_path: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    """Preflight, then call an existing trainer. Never computes audit metrics."""
    manifest = load_manifest(manifest_path)
    candidate = find_candidate(manifest, candidate_id)
    if candidate["status"] != "prepared":
        raise ValueError(f"candidate must be prepared, got {candidate['status']}")
    cfg, dataset, trainer = _candidate_config(manifest, candidate)
    if trainer.chat_date:
        os.environ["DP_LLM_CHAT_DATE"] = trainer.chat_date

    is_ddp = trainer.backend == "repo_train_dp_ddp"
    rank = int(os.environ.get("RANK", "0"))
    distributed = None
    if is_ddp:
        import torch
        import torch.distributed as dist

        distributed = dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    preflight_error = None
    if rank == 0:
        try:
            _run_preflight(manifest, candidate, cfg, dataset, trainer)
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
    if distributed is not None:
        messages = [preflight_error]
        distributed.broadcast_object_list(messages, src=0)
        preflight_error = messages[0]
    if preflight_error:
        if rank == 0:
            manifest = load_manifest(manifest_path)
            failed = find_candidate(manifest, candidate_id)
            failed["status"] = "preflight_failed"
            failed["error"] = preflight_error
            write_manifest(manifest_path, manifest)
        if distributed is not None:
            distributed.destroy_process_group()
        raise RuntimeError(preflight_error)

    if rank == 0:
        manifest = load_manifest(manifest_path)
        running = find_candidate(manifest, candidate_id)
        running["status"] = "running"
        write_manifest(manifest_path, manifest)

    result = None
    privacy = manifest["privacy"]
    if trainer.backend == "repo_train_dp":
        from finetune.train_dp import train_dp

        result = train_dp(cfg, epsilon=float(privacy["epsilon"]), pilot=False,
                          seed=int(manifest["environment"]["seed"]))
    elif trainer.backend == "repo_train_dp_ddp":
        from finetune.train_dp_ddp import train_dp_ddp

        train_dp_ddp(cfg, epsilon=float(privacy["epsilon"]), pilot=False,
                     seed=int(manifest["environment"]["seed"]))
        if rank == 0:
            result = _discover_result(candidate["trainer_results_dir"])
    else:
        from finetune.train_baseline import train_baseline

        result = train_baseline(cfg, pilot=False)

    if rank == 0 and result is not None:
        record_training_result(manifest_path, candidate_id, result)
    if distributed is not None:
        distributed.barrier()
        distributed.destroy_process_group()
    return result

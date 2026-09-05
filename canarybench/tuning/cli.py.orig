"""Cluster-friendly command line interface for tuning manifests."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .config import DatasetConfig, PrivacyConfig, SearchSpace, TrainerConfig
from .runner import run_repo_candidate
from .search import (
    activate_candidate,
    load_manifest,
    paper_table,
    prepare_search,
    record_validation,
    select_search,
)


def _model_defaults(model: str):
    from finetune.config import MODELS

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choices: {sorted(MODELS)}")
    return MODELS[model]


def _prepare(args) -> int:
    cfg = _model_defaults(args.model)
    data_root = Path(args.dataset_root).resolve()
    privacy = PrivacyConfig(
        mode=args.privacy_mode,
        epsilon=args.epsilon if args.privacy_mode == "dp" else None,
        delta=args.delta if args.privacy_mode == "dp" else None,
        accountant=args.accountant if args.privacy_mode == "dp" else None,
        max_grad_norm=cfg.max_grad_norm if args.privacy_mode == "dp" else None,
    )
    dataset = DatasetConfig(
        root=str(data_root),
        train_path=str(data_root / "combined_train.jsonl"),
        validation_path=str(data_root / "combined_val.jsonl"),
        member_path=str(data_root / "member_canaries.jsonl"),
        reference_path=str(data_root / "attack_eval.jsonl"),
        registry_path=str(data_root / "canary_registry.json"),
        validation_sensitivity=args.validation_sensitivity,
    )
    tokenizer_revision = args.tokenizer_revision or args.model_revision
    trainer = TrainerConfig(
        backend=args.trainer_backend,
        world_size=args.world_size,
        physical_batch_size=args.physical_batch_size or cfg.dp_physical_batch_size,
        model_revision=args.model_revision,
        tokenizer_revision=tokenizer_revision,
        chat_date=args.chat_date,
        python_executable=args.python_executable,
    )
    manifest = prepare_search(
        model=args.model,
        privacy=privacy,
        search_space=SearchSpace(
            learning_rate=args.learning_rates,
            epochs=args.epochs,
            global_batch_size=args.global_batch_sizes,
        ),
        dataset=dataset,
        trainer=trainer,
        output_dir=args.output_dir,
        base_batch_size=args.trainer_batch_size or cfg.batch_size,
        lora={
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": 0.0 if args.privacy_mode == "dp" else cfg.lora_dropout,
            "target_modules": cfg.target_modules,
        },
        seed=args.seed,
        search_policy=args.search_policy,
    )
    manifest_path = Path(args.output_dir).resolve() / "search_manifest.json"
    print(f"Manifest: {manifest_path}")
    for candidate in manifest["search"]["candidates"]:
        print(f"{candidate['status']}: {shlex.join(candidate['command'])}")
    return 0


def _evaluate_candidate(args) -> int:
    with open(args.metrics_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else payload.get("epochs", [payload])
    for item in records:
        record_validation(
            args.manifest,
            args.candidate_id,
            epoch=int(item["epoch"]),
            checkpoint=item["checkpoint"],
            metrics=item.get("metrics", item.get("validation", {})),
        )
    print(f"Recorded {len(records)} validation-only epoch result(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m canarybench.tuning")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="write a search manifest; launch nothing")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--privacy-mode", choices=("none", "dp"), required=True)
    prepare.add_argument("--epsilon", type=float)
    prepare.add_argument("--delta", type=float, default=1e-5)
    prepare.add_argument("--accountant", default="rdp")
    prepare.add_argument("--learning-rates", type=float, nargs="+", default=[2e-5, 1e-5])
    prepare.add_argument("--epochs", type=int, nargs="+", default=[20])
    prepare.add_argument("--global-batch-sizes", type=int, nargs="+", default=[128])
    prepare.add_argument("--search-policy", choices=("minimal", "grid"), default="minimal")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--validation-sensitivity", choices=("public", "non-sensitive", "private"), default="private")
    prepare.add_argument("--trainer-backend", choices=("repo_train_dp", "repo_train_dp_ddp", "repo_train_baseline"), required=True)
    prepare.add_argument("--world-size", type=int, default=1)
    prepare.add_argument("--trainer-batch-size", type=int)
    prepare.add_argument("--physical-batch-size", type=int)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--tokenizer-revision")
    prepare.add_argument("--chat-date")
    prepare.add_argument("--python-executable", default=sys.executable)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=_prepare)

    run = commands.add_parser("run-candidate", help="preflight then invoke the configured existing trainer")
    run.add_argument("--manifest", required=True)
    run.add_argument("--candidate-id", required=True)
    run.set_defaults(func=lambda args: (run_repo_candidate(args.manifest, args.candidate_id), 0)[1])

    activate = commands.add_parser("activate-candidate", help="explicitly approve a deferred minimal-policy candidate")
    activate.add_argument("--manifest", required=True)
    activate.add_argument("--candidate-id", required=True)
    activate.set_defaults(func=lambda args: (activate_candidate(args.manifest, args.candidate_id), 0)[1])

    evaluate = commands.add_parser("evaluate-candidate", help="ingest validation-only metrics emitted by the established evaluator")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--candidate-id", required=True)
    evaluate.add_argument("--metrics-json", required=True)
    evaluate.set_defaults(func=_evaluate_candidate)

    select = commands.add_parser("select", help="select across completed candidates and epochs")
    select.add_argument("--manifest", required=True)
    select.set_defaults(func=lambda args: (print(json.dumps(select_search(args.manifest), indent=2)), 0)[1])

    table = commands.add_parser("table", help="render selected.json files as a paper utility table")
    table.add_argument("selected", nargs="+")
    def _table(args):
        payloads = []
        for path in args.selected:
            with open(path, encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        print(paper_table(payloads))
        return 0
    table.set_defaults(func=_table)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))

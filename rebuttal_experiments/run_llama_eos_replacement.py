"""Fresh EOS-supervised Llama-3.2-1B non-DP replacement baseline."""

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("DP_LLM_CHAT_DATE", "31 Aug 2026")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

import torch
from canarybench.prompt_format import build_generation_prompt, build_supervised_text
from config import MODELS, require_dataset_paths
from dataset import MultiTaskDataset
import train_baseline as baseline

OUT = ROOT / "results/eos_fixed/non_dp/llama3_2_1b_it_eos_supervised_replacement"
DATA = ROOT / "instruction_recall_dataset/dataset/processed_v2"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


cfg = replace(MODELS["llama3_2_1b_it"])
cfg.model_name = str(SNAPSHOT)
cfg.train_path = str(DATA / "combined_train.jsonl")
cfg.val_path = str(DATA / "combined_val.jsonl")
cfg.canary_path = str(DATA / "member_canaries.jsonl")
cfg.reference_path = str(DATA / "attack_eval.jsonl")
cfg.canary_registry_path = str(DATA / "canary_registry.json")
cfg.output_base = str(OUT / "checkpoints")
cfg.results_dir = str(OUT / "train_results")
cfg.num_epochs = 8
cfg.learning_rate = 2e-5
cfg.batch_size = 8
cfg.grad_accum = 8
cfg.max_seq_length = 256
cfg.logging_steps = 100
cfg.eval_canaries = False
require_dataset_paths(cfg)

if (OUT / "checkpoints/llama3_2_1b_it/baseline/final/adapter_model.safetensors").exists():
    raise RuntimeError(f"Refusing to overwrite completed replacement: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = baseline._load_tokenizer(str(SNAPSHOT), "", {"local_files_only": True})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

records = []
seen = set()
with open(cfg.train_path, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        task = record.get("task")
        if task in {"classification", "nli", "pii_detection"} and task not in seen:
            records.append(record)
            seen.add(task)
        if len(records) == 3:
            break

checks = []
for record in records:
    prompt, _ = build_supervised_text(tokenizer, record)
    dataset_path = OUT / f".preflight_{record['task']}.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record, group=record.get("group", "A"))) + "\n")
    item = MultiTaskDataset(str(dataset_path), tokenizer, cfg.max_seq_length)[0]
    supervised_eos = (
        (item["input_ids"] == tokenizer.eos_token_id)
        & (item["attention_mask"] == 1)
        & (item["labels"] != -100)
    ).nonzero(as_tuple=False).flatten().tolist()
    check = {
        "task": record["task"],
        "train_eval_prompt_match": prompt
        == build_generation_prompt(tokenizer, record["input"]),
        "supervised_eos_positions": supervised_eos,
        "supervised_eos_present": bool(supervised_eos),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    if not check["train_eval_prompt_match"] or not check["supervised_eos_present"]:
        raise RuntimeError(f"Preflight failed: {check}")
    checks.append(check)
    dataset_path.unlink()

train_ds = MultiTaskDataset(cfg.train_path, tokenizer, cfg.max_seq_length)
usable = len(train_ds)
steps = math.ceil(usable / 64)
cfg.eval_steps = steps
cfg.save_steps = steps
del train_ds

OriginalTrainingArguments = baseline.TrainingArguments


def keep_all_epochs(*args, **kwargs):
    kwargs["save_total_limit"] = 20
    return OriginalTrainingArguments(*args, **kwargs)


baseline.TrainingArguments = keep_all_epochs
paths = {
    "train": cfg.train_path,
    "val": cfg.val_path,
    "member": cfg.canary_path,
    "reference": cfg.reference_path,
    "registry": cfg.canary_registry_path,
}
packages = {}
for name in [
    "torch", "transformers", "peft", "accelerate", "datasets", "numpy",
    "scipy", "scikit-learn", "sentencepiece", "protobuf",
]:
    try:
        packages[name] = version(name)
    except Exception as exc:
        packages[name] = f"unavailable:{type(exc).__name__}"

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
diff = subprocess.check_output(
    ["git", "diff", "--", "canarybench/prompt_format.py", "finetune/dataset.py",
     "tests/test_prompt_format.py"],
    cwd=ROOT,
)
manifest = {
    "status": "preflight_complete_training_starting",
    "replaces": str(ROOT / "results/eos_fixed/non_dp/llama3_2_1b_it"),
    "replacement_reason": "target EOS/EOT was masked because pad_token_id equaled eos_token_id",
    "privacy_mode": "none",
    "model_key": "llama3_2_1b_it",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "model_revision": SNAPSHOT.name,
    "tokenizer_revision": SNAPSHOT.name,
    "git_commit": commit,
    "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "seed": 42,
    "chat_date": os.environ["DP_LLM_CHAT_DATE"],
    "dataset_paths": paths,
    "dataset_sha256": {key: file_hash(path) for key, path in paths.items()},
    "train_records_usable": usable,
    "steps_per_epoch": steps,
    "protocol": {
        "epochs": 8,
        "per_device_batch": 8,
        "gradient_accumulation": 8,
        "effective_batch": 64,
        "learning_rate": 2e-5,
        "max_seq_length": 256,
        "checkpoint_selection": "validation_utility_only",
        "test_set_touched": False,
    },
    "packages": packages,
    "tokenizer_checks": checks,
    "test_set_touched": False,
}
with open(OUT / "run_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

baseline.train_baseline(cfg, pilot=False)
manifest["status"] = "training_complete_pending_validation"
with open(OUT / "run_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

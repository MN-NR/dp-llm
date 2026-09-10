"""CanaryBench dose-response experiment: Gemma-3-1B-it, non-DP, seed=42.

Uses the EXACT same corrected EOS-supervised training pipeline and
hyperparameters as the established Gemma-3 non-DP baseline
(results/eos_fixed/non_dp/gemma3_1b/ -- run_manifest.json's "protocol":
epochs=8, lr=2e-5, batch=8/grad_accum=8, max_seq_length=256, lora r=16
alpha=32 dropout=0.05, target_modules incl. gate/up/down_proj -- identical
overrides to the Llama non-DP replacement, just a different base model).
No hyperparameter is tuned here. The only experimental change is the
canary set: six frequency tiers (1x/2x/5x/10x/20x/50x, 800 unique
canaries, 8,450 insertions) instead of the original three tiers (1x/10x/
50x, 770 unique, 8,000 insertions).

Known, disclosed provenance difference (same situation as the Llama
non-DP dose-response run): Group A/B here come from
instruction_recall_dataset/dataset/processed/, not processed_v2/ (which
the eos_fixed/gemma3_1b baseline trained on -- both are independent draws
from the same build script/seed/source datasets, same task composition
and counts). Internally self-consistent across all 8 epochs of this run;
not byte-for-byte comparable to that baseline's absolute utility numbers.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from canarybench.prompt_format import build_generation_prompt, build_supervised_text
from config import MODELS, require_dataset_paths
from dataset import MultiTaskDataset
import train_baseline as baseline

OUT = ROOT / "results/dose_response/gemma3_1b/non_dp/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--google--gemma-3-1b-it/snapshots/"
    "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
FREQUENCY_TIERS = (1, 2, 5, 10, 20, 50)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


cfg = replace(MODELS["gemma3_1b"])
cfg.model_name = str(SNAPSHOT)
cfg.train_path = str(DATA / "combined_train.jsonl")
cfg.val_path = str(DATA / "combined_val.jsonl")
cfg.canary_path = str(DATA / "member_canaries.jsonl")
cfg.reference_path = str(DATA / "reference_canaries.jsonl")
cfg.canary_registry_path = str(DATA / "canary_registry.json")
cfg.output_base = str(OUT / "checkpoints")
cfg.results_dir = str(OUT / "train_results")
cfg.num_epochs = 8
cfg.learning_rate = 2e-5
cfg.batch_size = 8
cfg.grad_accum = 8
cfg.max_seq_length = 256
cfg.lora_dropout = 0.05
cfg.logging_steps = 100
cfg.eval_canaries = False
# Retain all 8 epoch checkpoints (save_total_limit=None) instead of the
# trainer's normal rotating limit of 2 -- required for per-epoch
# validation-based checkpoint selection below.
cfg.keep_all_checkpoints = True
require_dataset_paths(cfg)

if (OUT / "checkpoints/gemma3_1b/baseline/final/adapter_model.safetensors").exists():
    raise RuntimeError(f"Refusing to overwrite completed run: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = baseline._load_tokenizer(str(SNAPSHOT), "", {"local_files_only": True})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# EOS/prompt preflight over every task type present in this dataset,
# including secret_recall (the new dose-response canaries).
#
# Checks that the LAST real (non-padding) token of the formatted training
# sequence is supervised (label != -100), rather than searching for
# tokenizer.eos_token_id specifically. Gemma-3's chat template ends a turn
# with <end_of_turn> (id 106), a DIFFERENT token from tokenizer.eos_token
# ("<eos>", id 1) -- the model's own generation_config.json lists both
# ([1, 106]) as valid stop tokens, but tokenizer.eos_token_id alone would
# miss the one actually used to end assistant turns. finetune/dataset.py's
# EOS-supervision fix masks by attention-mask POSITION, not token value, so
# it already generalizes correctly across models; this check verifies that
# directly instead of assuming Llama's eos_token happens to equal its
# turn-ending token (confirmed separately, but not a safe assumption to
# hardcode for other models).
records = []
seen = set()
with open(cfg.train_path, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        task = record.get("task")
        if task in {"classification", "nli", "pii_detection", "secret_recall"} and task not in seen:
            records.append(record)
            seen.add(task)
        if len(records) == 4:
            break

checks = []
for record in records:
    prompt, _ = build_supervised_text(tokenizer, record)
    dataset_path = OUT / f".preflight_{record['task']}.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record, group=record.get("group", "A"))) + "\n")
    item = MultiTaskDataset(str(dataset_path), tokenizer, cfg.max_seq_length)[0]
    last_real_idx = int(item["attention_mask"].nonzero(as_tuple=False).max())
    last_token_id = int(item["input_ids"][last_real_idx])
    check = {
        "task": record["task"],
        "train_eval_prompt_match": prompt
        == build_generation_prompt(tokenizer, record["input"]),
        "last_real_token_id": last_token_id,
        "last_real_token_decoded": tokenizer.decode([last_token_id]),
        "supervised_last_token": bool(item["labels"][last_real_idx] != -100),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    if not check["train_eval_prompt_match"] or not check["supervised_last_token"]:
        raise RuntimeError(f"Preflight failed: {check}")
    checks.append(check)
    dataset_path.unlink()

train_ds = MultiTaskDataset(cfg.train_path, tokenizer, cfg.max_seq_length)
usable = len(train_ds)
steps = math.ceil(usable / 64)
cfg.eval_steps = steps
cfg.save_steps = steps
del train_ds

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
     "tests/test_prompt_format.py", "canarybench/evaluate.py", "metrics/privacy/memorization.py"],
    cwd=ROOT,
)
manifest = {
    "status": "preflight_complete_training_starting",
    "experiment": "canarybench_dose_response",
    "baseline_reference": str(ROOT / "results/eos_fixed/non_dp/gemma3_1b"),
    "dataset_provenance_note": (
        "Group A/B sourced from instruction_recall_dataset/dataset/processed/ "
        "(unchanged), NOT processed_v2/ (which the gemma3_1b eos_fixed baseline "
        "trained on). Internally self-consistent across all 8 epochs of this "
        "run; not byte-for-byte comparable to that baseline's absolute utility "
        "numbers. See module docstring."
    ),
    "privacy_mode": "none",
    "model_key": "gemma3_1b",
    "base_model": "google/gemma-3-1b-it",
    "model_revision": SNAPSHOT.name,
    "tokenizer_revision": SNAPSHOT.name,
    "git_commit": commit,
    "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "seed": 42,
    "dataset_paths": paths,
    "dataset_sha256": {key: file_hash(path) for key, path in paths.items()},
    "train_records_usable": usable,
    "steps_per_epoch": steps,
    "frequency_tiers": list(FREQUENCY_TIERS),
    "protocol": {
        "epochs": 8,
        "per_device_batch": 8,
        "gradient_accumulation": 8,
        "effective_batch": 64,
        "learning_rate": 2e-5,
        "max_seq_length": 256,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": cfg.target_modules,
        "checkpoint_selection": "validation_utility_only",
        "checkpoint_selection_formula": "(ag_news_greedy_accuracy + mnli_greedy_accuracy + pii_unique_f1) / 3",
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

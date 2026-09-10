"""CanaryBench dose-response experiment: Llama-3.2-1B-Instruct, non-DP, seed=42.

Uses the EXACT same corrected EOS-supervised Llama training pipeline and
hyperparameters as results/eos_fixed/non_dp/llama3_2_1b_it_eos_supervised_replacement
(see rebuttal_experiments/run_llama_eos_replacement.py, which this mirrors
line-for-line except for dataset/output paths). No hyperparameter is tuned
here. The only experimental change is the canary set: six frequency tiers
(1x/2x/5x/10x/20x/50x, 800 unique canaries, 8,450 insertions) built by
instruction_recall_dataset/build_dose_response_canaries.py, instead of the
original three tiers (1x/10x/50x, 770 unique, 8,000 insertions).

Known, disclosed provenance difference: Group A/B (utility + PII detection)
here come from instruction_recall_dataset/dataset/processed/ (unchanged),
not dataset/processed_v2/ (which is what the EOS-replacement baseline
trained on -- both are independent draws from the same build script/seed/
source datasets, same task composition and counts). This does not affect
internal validity of the frequency-dose-response comparison, since all 8
epoch checkpoints of THIS run are trained and validated against the same
single frozen dataset throughout -- but it does mean this run's absolute
utility numbers are not byte-for-byte comparable to the EOS-replacement
baseline's utility numbers. Flagged here and in run_manifest.json rather
than silently training on a different dataset than what was asked for.
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

os.environ.setdefault("DP_LLM_CHAT_DATE", "31 Aug 2026")
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

OUT = ROOT / "results/dose_response/llama3_2_1b/non_dp/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)
FREQUENCY_TIERS = (1, 2, 5, 10, 20, 50)


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
cfg.reference_path = str(DATA / "reference_canaries.jsonl")
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
# Retain all 8 epoch checkpoints (save_total_limit=None) instead of the
# trainer's normal rotating limit of 2 -- required for per-epoch
# validation-based checkpoint selection below.
cfg.keep_all_checkpoints = True
require_dataset_paths(cfg)

if (OUT / "checkpoints/llama3_2_1b_it/baseline/final/adapter_model.safetensors").exists():
    raise RuntimeError(f"Refusing to overwrite completed run: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = baseline._load_tokenizer(str(SNAPSHOT), "", {"local_files_only": True})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# EOS/prompt preflight over every task type present in this dataset,
# including secret_recall (the new dose-response canaries) -- the original
# replacement baseline's preflight only covered classification/nli/
# pii_detection since canaries used the identical code path; we check
# secret_recall explicitly too since it's this experiment's whole point.
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
    "baseline_reference": str(ROOT / "results/eos_fixed/non_dp/llama3_2_1b_it_eos_supervised_replacement"),
    "dataset_provenance_note": (
        "Group A/B sourced from instruction_recall_dataset/dataset/processed/ "
        "(unchanged), NOT processed_v2/ (which the EOS-replacement baseline "
        "trained on). Internally self-consistent across all 8 epochs of this "
        "run; not byte-for-byte comparable to the baseline's absolute utility "
        "numbers. See module docstring."
    ),
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
    "frequency_tiers": list(FREQUENCY_TIERS),
    "protocol": {
        "epochs": 8,
        "per_device_batch": 8,
        "gradient_accumulation": 8,
        "effective_batch": 64,
        "learning_rate": 2e-5,
        "max_seq_length": 256,
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

"""CanaryBench dose-response experiment: Gemma-3-1B-it, CLIPPING-ONLY
mechanistic ablation (C=1, noise_multiplier=0), seed=42.

Same purpose and mechanism as run_llama_clip_only_dose_response.py (see
that file's docstring for the full rationale) applied to Gemma-3: isolates
clipping's contribution from noise's to the flattened non-DP -> DP eps=5
extraction curve (results/dose_response/gemma3_1b/{non_dp,dp_eps5}/seed42/)
by running finetune/train_dp.py's Opacus path with per-sample clipping to
max_grad_norm=cfg.max_grad_norm (=1.0, unchanged config.py default) but
noise_multiplier forced to 0.0 instead of calibrated.

Same hyperparameters as run_gemma3_dp_eps5_dose_response.py otherwise
(epochs=3, batch_size=8, grad_accum=8, dp_physical_batch_size=2,
learning_rate=1e-5) -- nothing tuned, only noise_multiplier changed.
"""

import hashlib
import json
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
import train_dp as train_dp_module
from train_dp import train_dp

OUT = ROOT / "results/dose_response/gemma3_1b/clip_only/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--google--gemma-3-1b-it/snapshots/"
    "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
FREQUENCY_TIERS = (1, 2, 5, 10, 20, 50)
NOISE_MULTIPLIER = 0.0
SEED = 42


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
cfg.eval_canaries = False
cfg.save_epoch_checkpoints = True
require_dataset_paths(cfg)

run_tag = f"clip_only_C{cfg.max_grad_norm}_seed{SEED}"
final_adapter = Path(cfg.output_base) / cfg.model_key / run_tag / "final" / "adapter_model.safetensors"
if final_adapter.exists():
    raise RuntimeError(f"Refusing to overwrite completed run: {final_adapter}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = train_dp_module._load_tokenizer(str(SNAPSHOT), "", {"local_files_only": True})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    "scipy", "scikit-learn", "sentencepiece", "protobuf", "opacus",
]:
    try:
        packages[name] = version(name)
    except Exception as exc:
        packages[name] = f"unavailable:{type(exc).__name__}"

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
diff = subprocess.check_output(
    ["git", "diff", "--", "canarybench/prompt_format.py", "finetune/dataset.py",
     "finetune/train_dp.py", "canarybench/evaluate.py", "metrics/privacy/memorization.py"],
    cwd=ROOT,
)
manifest = {
    "status": "preflight_complete_training_starting",
    "experiment": "canarybench_dose_response_clipping_only_ablation",
    "ablation_purpose": (
        "Isolate clipping's contribution vs noise's to the DP eps=5 flattened "
        "extraction curve. Same Opacus path (per-sample gradient clipping to "
        "max_grad_norm=1.0) as the DP eps=5 run, but noise_multiplier forced "
        "to 0.0 instead of calibrated -- i.e. NO privacy guarantee here, this "
        "is a mechanism probe, not a private training run."
    ),
    "privacy_mode": "clipping_only",
    "target_epsilon": None,
    "noise_multiplier": NOISE_MULTIPLIER,
    "max_grad_norm": cfg.max_grad_norm,
    "compare_against": [
        str(ROOT / "results/dose_response/gemma3_1b/non_dp/seed42"),
        str(ROOT / "results/dose_response/gemma3_1b/dp_eps5/seed42"),
    ],
    "model_key": "gemma3_1b",
    "base_model": "google/gemma-3-1b-it",
    "model_revision": SNAPSHOT.name,
    "tokenizer_revision": SNAPSHOT.name,
    "git_commit": commit,
    "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "seed": SEED,
    "dataset_paths": paths,
    "dataset_sha256": {key: file_hash(path) for key, path in paths.items()},
    "frequency_tiers": list(FREQUENCY_TIERS),
    "protocol": {
        "epochs": cfg.num_epochs,
        "per_device_batch": cfg.batch_size,
        "gradient_accumulation": cfg.grad_accum,
        "effective_batch": cfg.batch_size * cfg.grad_accum,
        "dp_physical_batch_size": cfg.dp_physical_batch_size,
        "learning_rate": cfg.learning_rate,
        "max_grad_norm": cfg.max_grad_norm,
        "max_seq_length": cfg.max_seq_length,
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

results = train_dp(cfg, epsilon=None, pilot=False, seed=SEED, noise_multiplier=NOISE_MULTIPLIER)
manifest["status"] = "training_complete_pending_validation"
manifest["final_epsilon_reported_by_opacus"] = results.get("final_epsilon")
manifest["train_result_summary"] = {
    k: results.get(k) for k in [
        "best_val_loss", "train_samples", "val_samples", "epochs",
        "sample_rate", "optimizer_steps", "train_runtime_hrs", "output_dir",
    ]
}
with open(OUT / "run_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

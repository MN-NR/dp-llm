"""CanaryBench dose-response experiment: Llama-3.2-1B-Instruct, DP (target eps=5.0),
seed=42.

No "EOS-corrected" DP baseline exists yet for Llama (unlike non-DP, where
run_llama_eos_replacement.py was the established reference) -- see the
research summary in-conversation. Per explicit user choice, this reuses the
ONLY prior Llama eps=5 DP recipe (finetune/config.py's llama3_2_1b_it
defaults, single-GPU finetune/train_dp.py, from the pre-EOS-fix run whose
results live at results/instruction_recall/llama3_2_1b_it_dp_eps5.0_*.json)
completely unchanged:
    epochs=3, batch_size=8, grad_accum=8 (logical batch 64),
    dp_physical_batch_size=8, learning_rate=1e-5, max_grad_norm=1.0,
    lora_r=16, lora_alpha=32 (lora_dropout hardcoded to 0.0 inside
    train_dp.py itself, required for Opacus -- unchanged from that file).
No hyperparameter is tuned here. The only experimental changes are (a) the
already-fixed finetune/dataset.py EOS-supervision masking (shared code,
applies automatically -- nothing to configure) and (b) the six-tier
dose-response canary ladder (1x/2x/5x/10x/20x/50x, 800 unique, 8,450
insertions) in place of the original three-tier canary set.

Same disclosed provenance note as run_llama_dose_response.py: Group A/B
here come from instruction_recall_dataset/dataset/processed/, the same
dataset the non-DP dose-response run used (and the ONLY prior Llama DP
eps=5 run also used dataset/processed/ via MODELS["llama3_2_1b_it"]'s
built-in default paths, so this is actually consistent with that
precedent, unlike the non-DP EOS-replacement baseline which used
processed_v2/).
"""

import hashlib
import json
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
import train_dp as train_dp_module
from train_dp import train_dp

OUT = ROOT / "results/dose_response/llama3_2_1b/dp_eps5/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)
FREQUENCY_TIERS = (1, 2, 5, 10, 20, 50)
TARGET_EPSILON = 5.0
SEED = 42


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
cfg.eval_canaries = False
# Every epoch needed for validation-utility checkpoint selection (only 3
# epochs total here, per the unchanged config.py/pre-fix-run recipe).
cfg.save_epoch_checkpoints = True
require_dataset_paths(cfg)

run_tag = f"dp_eps{TARGET_EPSILON}_seed{SEED}"
final_adapter = Path(cfg.output_base) / cfg.model_key / run_tag / "final" / "adapter_model.safetensors"
if final_adapter.exists():
    raise RuntimeError(f"Refusing to overwrite completed run: {final_adapter}")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = train_dp_module._load_tokenizer(str(SNAPSHOT), "", {"local_files_only": True})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Same EOS/prompt preflight as the non-DP dose-response run -- dataset.py's
# EOS-supervision fix is shared code, so this should pass identically here.
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
    "experiment": "canarybench_dose_response",
    "privacy_mode": "dp",
    "target_epsilon": TARGET_EPSILON,
    "delta": cfg.delta,
    "hyperparameter_provenance": (
        "Reused unchanged from finetune/config.py's llama3_2_1b_it defaults "
        "and the only prior Llama eps=5 DP run "
        "(results/instruction_recall/llama3_2_1b_it_dp_eps5.0_20260429_063351.json), "
        "which predates the EOS-supervision fix. No eos_fixed/ DP baseline exists "
        "for Llama (only VaultGemma, via a swept/DDP recipe not calibrated for "
        "Llama). User explicitly chose to reuse the pre-fix hyperparameters "
        "rather than adapt VaultGemma's recipe or run a fresh hp-search."
    ),
    "model_key": "llama3_2_1b_it",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "model_revision": SNAPSHOT.name,
    "tokenizer_revision": SNAPSHOT.name,
    "git_commit": commit,
    "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "seed": SEED,
    "chat_date": os.environ["DP_LLM_CHAT_DATE"],
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

results = train_dp(cfg, epsilon=TARGET_EPSILON, pilot=False, seed=SEED)
manifest["status"] = "training_complete_pending_validation"
manifest["final_epsilon"] = results.get("final_epsilon")
manifest["noise_multiplier"] = results.get("noise_multiplier")
manifest["train_result_summary"] = {
    k: results.get(k) for k in [
        "best_val_loss", "train_samples", "val_samples", "epochs",
        "sample_rate", "optimizer_steps", "train_runtime_hrs", "output_dir",
    ]
}
with open(OUT / "run_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

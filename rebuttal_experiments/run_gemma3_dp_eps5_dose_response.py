"""CanaryBench dose-response experiment: Gemma-3-1B-it, DP (target eps=5.0),
seed=42.

Same situation and same user-chosen policy as
run_llama_dp_eps5_dose_response.py: no EOS-corrected DP baseline exists for
Gemma-3 either (results/eos_fixed/ only has a corrected DP recipe for
VaultGemma, via a swept/DDP recipe not calibrated for this model). Reuses
the pre-fix Gemma-3 DP recipe unchanged: finetune/config.py's gemma3_1b
defaults, single-GPU finetune/train_dp.py (the same recipe referenced by
rebuttal_experiments/results/matched_all_eps/gemma3_1b_..._dp_eps5.0_...json,
whose checkpoint predates the EOS fix):
    epochs=3, batch_size=8, grad_accum=8 (logical batch 64),
    dp_physical_batch_size=2, learning_rate=1e-5, max_grad_norm=1.0,
    lora_r=16, lora_alpha=32 (lora_dropout hardcoded to 0.0 inside
    train_dp.py itself, required for Opacus).
No hyperparameter is tuned here. The only experimental changes are (a) the
already-fixed finetune/dataset.py EOS-supervision masking (shared code) and
(b) the six-tier dose-response canary ladder (1x/2x/5x/10x/20x/50x, 800
unique, 8,450 insertions) in place of the original three-tier canary set.

Same disclosed provenance note: Group A/B come from
instruction_recall_dataset/dataset/processed/, matching what the only
prior Gemma-3 DP eps=5 run also used via MODELS["gemma3_1b"]'s built-in
default paths.
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

OUT = ROOT / "results/dose_response/gemma3_1b/dp_eps5/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--google--gemma-3-1b-it/snapshots/"
    "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
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

# EOS/prompt preflight -- checks that the LAST real (non-padding) token of
# the formatted training sequence is supervised, rather than searching for
# tokenizer.eos_token_id specifically. Gemma-3's chat template ends a turn
# with <end_of_turn> (id 106), a DIFFERENT token from tokenizer.eos_token
# ("<eos>", id 1) -- see run_gemma3_dose_response.py's non-DP counterpart
# for the full story (a real bug was found and fixed there in the
# validation-loop's .generate() call; this script's own validation loop,
# in postprocess_gemma3_dp_eps5_dose_response.py, has the same fix applied
# from the start).
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
    "experiment": "canarybench_dose_response",
    "privacy_mode": "dp",
    "target_epsilon": TARGET_EPSILON,
    "delta": cfg.delta,
    "hyperparameter_provenance": (
        "Reused unchanged from finetune/config.py's gemma3_1b defaults and the "
        "only prior Gemma-3 eps=5 DP run "
        "(rebuttal_experiments/results/matched_all_eps/gemma3_1b_..._dp_eps5.0_...json), "
        "which predates the EOS-supervision fix. No eos_fixed/ DP baseline exists "
        "for Gemma-3 (only VaultGemma, via a swept/DDP recipe not calibrated for "
        "Gemma-3). Same policy the user chose for the Llama DP dose-response run, "
        "applied consistently here."
    ),
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

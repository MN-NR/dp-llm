"""Validation-only checkpoint selection + CanaryBench audit for the Llama-3.2-1B
clipping-only mechanistic ablation (results/dose_response/llama3_2_1b/clip_only/seed42/).

Adapted from postprocess_llama_dp_eps5_dose_response.py -- identical U_val
formula, tie-break rule, and canarybench.evaluate.evaluate() call. Epoch
checkpoints come from finetune/train_dp.py's own
output_dir/epoch_checkpoints/epoch_XX layout (run_tag here is
"clip_only_C{max_grad_norm}_seed{seed}", set by train_dp.py's
noise_multiplier parameter -- see that file and
run_llama_clip_only_dose_response.py).
"""

import csv
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DP_LLM_CHAT_DATE", "31 Aug 2026")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path("/home/noora/dp-llm/dp-llm")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

OUT = ROOT / "results/dose_response/llama3_2_1b/clip_only/seed42"
DATA = ROOT / "instruction_recall_dataset/dataset/dose_response"
SNAPSHOT = Path(
    "/home/noora/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    "9213176726f574b556790deb65791e0c5aa438b6"
)
FREQUENCY_TIERS = (1, 2, 5, 10, 20, 50)
N_EXTRACTION_TARGETS = 20

RUN_TAG = "clip_only_C1.0_seed42"
TRAIN_DIR = OUT / "checkpoints/llama3_2_1b_it" / RUN_TAG
EPOCH_DIR = TRAIN_DIR / "epoch_checkpoints"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from canarybench.prompt_format import build_generation_prompt
from finetune.test_model import relaxed_pii_scores

if not (TRAIN_DIR / "final/adapter_model.safetensors").exists():
    raise RuntimeError("training ended without final adapter -- run_llama_clip_only_dose_response.py first")

manifest_path = OUT / "run_manifest.json"
manifest = json.load(open(manifest_path))
num_epochs = manifest["protocol"]["epochs"]
epoch_dirs = {epoch: EPOCH_DIR / f"epoch_{epoch:02d}" for epoch in range(1, num_epochs + 1)}
missing = [e for e, d in epoch_dirs.items() if not (d / "adapter_model.safetensors").exists()]
if missing:
    raise RuntimeError(f"missing epoch checkpoints: {missing}")

val_records = [json.loads(x) for x in open(DATA / "combined_val.jsonl", encoding="utf-8") if x.strip()]
by_task = {t: [r for r in val_records if r.get("task") == t] for t in ["classification", "nli", "pii_detection"]}
if any(len(v) != 1000 for v in by_task.values()):
    raise RuntimeError(f"unexpected val counts: { {k: len(v) for k, v in by_task.items()} }")


def _norm(text):
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _clean_first_line(text):
    return text.strip().splitlines()[0].strip().strip("`").strip('"\'').rstrip(".:;!,").strip()


def _generate(model, tokenizer, rows, max_new_tokens, batch_size):
    out = []
    tokenizer.padding_side = "left"
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [build_generation_prompt(tokenizer, r["input"]) for r in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [_clean_first_line(x) for x in decoded]
    return out


def validate_checkpoint(checkpoint_path, epoch_index):
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.bfloat16, local_files_only=True).cuda()
    model = PeftModel.from_pretrained(model, checkpoint_path).eval()

    classification_gen = _generate(model, tokenizer, by_task["classification"], 12, 16)
    nli_gen = _generate(model, tokenizer, by_task["nli"], 12, 16)
    pii_gen = _generate(model, tokenizer, by_task["pii_detection"], 80, 8)
    classification_norm = list(map(_norm, classification_gen))
    nli_norm = list(map(_norm, nli_gen))

    ag_accuracy = sum(x == _norm(r["target"]) for x, r in zip(classification_norm, by_task["classification"])) / 1000
    mnli_accuracy = sum(x == _norm(r["target"]) for x, r in zip(nli_norm, by_task["nli"])) / 1000
    ag_invalid = sum(x not in {"world", "sports", "business", "sci/tech"} for x in classification_norm) / 1000
    nli_invalid = sum(x not in {"entailment", "neutral", "contradiction"} for x in nli_norm) / 1000
    pii_metrics = relaxed_pii_scores(
        [{"target": r["target"], "generated": x} for r, x in zip(by_task["pii_detection"], pii_gen)]
    )
    utility_score = (ag_accuracy + mnli_accuracy + pii_metrics["pii_unique_f1"]) / 3

    result = {
        "checkpoint": str(checkpoint_path),
        "decoding": "greedy",
        "selection_uses_privacy_metrics": False,
        "classification": {"accuracy": ag_accuracy, "invalid_generation_rate": ag_invalid, "n": 1000},
        "nli": {"accuracy": mnli_accuracy, "invalid_generation_rate": nli_invalid, "n": 1000},
        "pii_detection": pii_metrics,
        "ag_news_greedy_accuracy": ag_accuracy,
        "mnli_greedy_accuracy": mnli_accuracy,
        "pii_unique_precision": pii_metrics["pii_unique_precision"],
        "pii_unique_recall": pii_metrics["pii_unique_recall"],
        "pii_unique_f1": pii_metrics["pii_unique_f1"],
        "pii_exact_match": pii_metrics["pii_unique_exact_match"],
        "repetition_fraction": pii_metrics["repetition_frac_examples_with_dupes"],
        "invalid_generation_rate": (ag_invalid + nli_invalid) / 2,
        "utility_score": utility_score,
        "epoch_index": epoch_index,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


validation_dir = OUT / "validation"
validation_dir.mkdir(exist_ok=True)
epoch_results = []
for epoch in range(1, num_epochs + 1):
    print(f"[validation] epoch {epoch}", flush=True)
    result = validate_checkpoint(epoch_dirs[epoch], epoch)
    epoch_results.append(result)
    json.dump(result, open(validation_dir / f"epoch_{epoch:02d}.json", "w"), indent=2)
json.dump(
    {"evaluated_epochs": list(range(1, num_epochs + 1)), "unavailable_epochs": [], "results": epoch_results},
    open(validation_dir / "all_epochs.json", "w"), indent=2,
)

best = sorted(
    epoch_results,
    key=lambda r: (-r["utility_score"], r["invalid_generation_rate"], r["repetition_fraction"], r["epoch_index"]),
)[0]
selected_dir = OUT / f"selected_checkpoint_epoch_{best['epoch_index']:02d}"
if selected_dir.exists():
    raise RuntimeError(f"selected checkpoint dir already exists: {selected_dir}")
shutil.copytree(Path(best["checkpoint"]), selected_dir)

selection = {
    "policy": (
        "maximize U_val = (AG News greedy accuracy + MNLI greedy accuracy + PII Unique F1) / 3; "
        "tie-break lower invalid rate, lower repetition, earlier epoch"
    ),
    "privacy_or_memorization_metrics_used": False,
    "evaluated_epochs": list(range(1, num_epochs + 1)),
    "unavailable_epochs": [],
    "selected_epoch": best["epoch_index"],
    "selected_checkpoint": str(selected_dir),
    "selected_metrics": best,
}
json.dump(selection, open(OUT / "checkpoint_selection.json", "w"), indent=2)
print(f"[audit] selected epoch {best['epoch_index']}", flush=True)

from canarybench.evaluate import evaluate

member_count = sum(1 for _ in open(DATA / "member_canaries.jsonl"))
reference_count = sum(1 for _ in open(DATA / "reference_canaries.jsonl"))
artifacts = SimpleNamespace(
    member_path=str(DATA / "member_canaries.jsonl"),
    reference_path=str(DATA / "reference_canaries.jsonl"),
)
audit_result = evaluate(
    str(SNAPSHOT), str(selected_dir), artifacts,
    device="cuda",
    max_member_samples=member_count,
    max_reference_samples=reference_count,
    reference_model_path=str(SNAPSHOT),
    run_wbc=True,
    run_extraction=True,
    n_extraction_targets=N_EXTRACTION_TARGETS,
    mask_prompt=False,
    frequency_tiers=FREQUENCY_TIERS,
).to_dict()
audit_result.update(
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    offline_enforced=True,
    checkpoint_selection=selection,
    frequency_tiers=list(FREQUENCY_TIERS),
    privacy_mode="clipping_only",
    max_grad_norm=manifest.get("max_grad_norm"),
    noise_multiplier=manifest.get("noise_multiplier"),
)

import numpy as np
from metrics.privacy.memorization import exposure_by_repetition

member_values = np.array([x["loss"] for x in audit_result["exposure"]["member_losses"]])
reference_values = np.array([x["loss"] for x in audit_result["exposure"]["reference_losses"]])
rng = np.random.default_rng(42)
bootstrap_gaps = np.array([
    rng.choice(reference_values, len(reference_values)).mean() - rng.choice(member_values, len(member_values)).mean()
    for _ in range(2000)
])
bootstrap_ci = {
    "estimate": float(reference_values.mean() - member_values.mean()),
    "ci_lower": float(np.percentile(bootstrap_gaps, 2.5)),
    "ci_upper": float(np.percentile(bootstrap_gaps, 97.5)),
    "confidence": 0.95,
    "n_bootstrap": 2000,
    "seed": 42,
}
audit_result["memorization_gap_bootstrap"] = bootstrap_ci
json.dump(audit_result, open(OUT / "canarybench_audit_full.json", "w"), indent=2)

exposure = audit_result["exposure"]
wbc = audit_result["wbc"]

# ---- Per-canary export ------------------------------------------------
member_losses = exposure["member_losses"]
member_exposures = exposure["member_exposures"]
member_wbc_scores = wbc["member_scores"]
assert len(member_losses) == len(member_exposures) == len(member_wbc_scores), (
    "member_losses/member_exposures/wbc member_scores must be same length and order "
    "(all three are computed from the same member_canaries list inside evaluate())"
)

extraction_outcome = {}
for tier_key, tier_data in audit_result["targeted_extraction"].items():
    for detail in tier_data["details"]:
        extraction_outcome[detail["canary_id"]] = int(detail["exact_hit"])

reference_loss_mean = exposure["reference_loss_mean"]
per_canary_rows = []
for loss_record, exposure_value, wbc_score in zip(member_losses, member_exposures, member_wbc_scores):
    canary_id = loss_record.get("id", "")
    per_canary_rows.append({
        "canary_id": canary_id,
        "frequency": loss_record.get("exposure"),
        "field_type": loss_record.get("field_type"),
        "loss": round(float(loss_record["loss"]), 6),
        "loss_based_memorization_score": round(reference_loss_mean - float(loss_record["loss"]), 6),
        "exposure": round(float(exposure_value), 6),
        "wbc_score": round(float(wbc_score), 6),
        "extracted": extraction_outcome.get(canary_id, ""),
    })

per_canary_csv = OUT / "per_canary_metrics.csv"
with open(per_canary_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(per_canary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_canary_rows)
with open(OUT / "per_canary_metrics.jsonl", "w", encoding="utf-8") as handle:
    for row in per_canary_rows:
        handle.write(json.dumps(row) + "\n")

# ---- Frequency-level summary table -------------------------------------
exposure_means_by_tier = exposure_by_repetition(member_losses, exposure["reference_losses"])
wbc_by_tier = {}
for row in per_canary_rows:
    wbc_by_tier.setdefault(row["frequency"], []).append(row["wbc_score"])

frequency_table = []
for tier in FREQUENCY_TIERS:
    tier_key = f"{tier}x"
    freq_stats = audit_result["frequency_breakdown"].get(tier_key, {})
    extraction_stats = audit_result["targeted_extraction"].get(tier_key, {})
    wbc_scores = wbc_by_tier.get(tier, [])
    frequency_table.append({
        "frequency": tier,
        "n": freq_stats.get("n", 0),
        "gap": freq_stats.get("gap"),
        "wbc_mean_score": round(float(np.mean(wbc_scores)), 6) if wbc_scores else None,
        "exposure_mean": round(exposure_means_by_tier.get(f"exposure_{tier}", 0.0), 6),
        "extraction_rate": extraction_stats.get("pii_extraction_rate"),
        "extraction_n_targets": extraction_stats.get("n_targets"),
        "extraction_num_reproduced_exactly": extraction_stats.get("num_reproduced_exactly"),
    })

non_zero_extraction = [row for row in frequency_table if (row["extraction_rate"] or 0) > 0]
f_star = min(row["frequency"] for row in non_zero_extraction) if non_zero_extraction else None

json.dump(
    {
        "frequency_table": frequency_table,
        "f_star": f_star,
        "f_star_definition": (
            "Lowest frequency tier (out of the 20 most-memorized/targeted canaries per tier) "
            "at which pii_extraction_rate first becomes non-zero. None if no tier extracted."
        ),
    },
    open(OUT / "frequency_summary.json", "w"), indent=2,
)


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


print("\nFrequency | N | gap | WBC(mean) | exposure | extraction_rate")
for row in frequency_table:
    print(
        f"{row['frequency']:>4}x | {row['n']:>4} | {_fmt(row['gap'])} | "
        f"{_fmt(row['wbc_mean_score'])} | {_fmt(row['exposure_mean'])} | {_fmt(row['extraction_rate'])}"
    )
print(f"\nEmpirical extraction threshold F* = {f_star}")

summary = {
    "selected_epoch": best["epoch_index"],
    "privacy_mode": "clipping_only",
    "max_grad_norm": manifest.get("max_grad_norm"),
    "noise_multiplier": manifest.get("noise_multiplier"),
    "validation_metrics": {
        k: best[k] for k in [
            "ag_news_greedy_accuracy", "mnli_greedy_accuracy", "pii_unique_precision",
            "pii_unique_recall", "pii_unique_f1", "pii_exact_match", "repetition_fraction",
            "invalid_generation_rate", "utility_score",
        ]
    },
    "member_loss": exposure["member_loss_mean"],
    "reference_loss": exposure["reference_loss_mean"],
    "memorization_gap": exposure["gap"],
    "memorization_gap_bootstrap": bootstrap_ci,
    "gap_by_frequency": audit_result["frequency_breakdown"],
    "wbc": {k: wbc[k] for k in ["wbc_auc", "auc_ci_lower", "auc_ci_upper", "loss_baseline_auc"]},
    "exposure": {k: exposure[k] for k in ["exposure_mean", "exposure_p50", "exposure_p95", "exposure_max"]},
    "targeted_greedy_extraction": audit_result["targeted_extraction"],
    "frequency_table": frequency_table,
    "f_star": f_star,
    "evaluated_epochs": list(range(1, num_epochs + 1)),
    "unavailable_epochs": [],
    "test_set_touched": False,
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

manifest.update(
    status="complete",
    selected_epoch=best["epoch_index"],
    selected_checkpoint=str(selected_dir),
    f_star=f_star,
    test_set_touched=False,
)
json.dump(manifest, open(manifest_path, "w"), indent=2)
print("\n[post] COMPLETE", OUT / "summary.json", flush=True)

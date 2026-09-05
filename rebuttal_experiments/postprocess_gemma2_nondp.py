"""Validation-only epoch selection and post-selection audit for Gemma-2-2B-IT."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from canarybench.evaluate import evaluate as audit_checkpoint
from canarybench.prompt_format import build_generation_prompt
from canarybench.tuning import record_validation, select_search
from finetune.test_model import relaxed_pii_scores


OUT = ROOT / "results/eos_fixed/non_dp/gemma_2b_it"
DATA = ROOT / "instruction_recall_dataset/dataset/processed_v2"
BASE = Path(
    "/home/noora/.cache/huggingface/hub/models--google--gemma-2-2b-it/"
    "snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
)
MANIFEST = OUT / "search_manifest.json"
CANDIDATE_ID = "lr_2e-05_ep_08_gb_64"
CANDIDATE_CHECKPOINTS = (
    OUT / "candidates" / CANDIDATE_ID / "checkpoints" / "gemma_2b_it" / "baseline"
)


def normalize_label(value: str) -> str:
    first = str(value or "").strip().splitlines()[0] if str(value or "").strip() else ""
    return re.sub(r"\s+", " ", first).strip().strip("`").strip().strip("\"'").rstrip(".:;!,").lower()


def load_validation():
    rows = [
        json.loads(line)
        for line in open(DATA / "combined_val.jsonl", encoding="utf-8")
        if line.strip()
    ]
    tasks = {
        task: [record for record in rows if record.get("task") == task]
        for task in ("classification", "nli", "pii_detection")
    }
    if any(len(records) != 1000 for records in tasks.values()):
        raise RuntimeError(
            "frozen validation task counts changed: "
            + repr({task: len(records) for task, records in tasks.items()})
        )
    return tasks


def generate(model, tokenizer, records, max_new_tokens, batch_size):
    generated = []
    tokenizer.padding_side = "left"
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [build_generation_prompt(tokenizer, record["input"]) for record in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        generated.extend(
            value.strip()
            for value in tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)
        )
    return generated


def evaluate_epoch(checkpoint: Path, epoch: int, tasks):
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    model = PeftModel.from_pretrained(model, checkpoint).eval()

    ag_raw = generate(model, tokenizer, tasks["classification"], 12, 16)
    mnli_raw = generate(model, tokenizer, tasks["nli"], 12, 16)
    pii_raw = generate(model, tokenizer, tasks["pii_detection"], 80, 8)
    ag = [normalize_label(value) for value in ag_raw]
    mnli = [normalize_label(value) for value in mnli_raw]
    ag_accuracy = sum(
        pred == normalize_label(record["target"])
        for pred, record in zip(ag, tasks["classification"])
    ) / len(ag)
    mnli_accuracy = sum(
        pred == normalize_label(record["target"])
        for pred, record in zip(mnli, tasks["nli"])
    ) / len(mnli)
    ag_invalid = sum(
        pred not in {"world", "sports", "business", "sci/tech"} for pred in ag
    ) / len(ag)
    mnli_invalid = sum(
        pred not in {"entailment", "neutral", "contradiction"} for pred in mnli
    ) / len(mnli)
    pii = relaxed_pii_scores([
        {"target": record["target"], "generated": pred}
        for record, pred in zip(tasks["pii_detection"], pii_raw)
    ])
    metrics = {
        "ag_news_accuracy": ag_accuracy,
        "mnli_accuracy": mnli_accuracy,
        "pii_unique_f1": pii["pii_unique_f1"],
        "pii_exact_match": pii["pii_unique_exact_match"],
        "invalid_rate": (ag_invalid + mnli_invalid) / 2,
        "repetition_fraction": pii["repetition_frac_examples_with_dupes"],
    }
    detail = {
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "decoding": "greedy",
        "classification": {"accuracy": ag_accuracy, "invalid_generation_rate": ag_invalid, "n": len(ag)},
        "nli": {"accuracy": mnli_accuracy, "invalid_generation_rate": mnli_invalid, "n": len(mnli)},
        "pii_detection": pii,
        **metrics,
        "u_val": (ag_accuracy + mnli_accuracy + pii["pii_unique_f1"]) / 3,
        "test_set_touched": False,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, detail


def epoch_checkpoints():
    found = []
    for checkpoint in CANDIDATE_CHECKPOINTS.glob("checkpoint-*"):
        state_path = checkpoint / "trainer_state.json"
        if not state_path.exists() or not (checkpoint / "adapter_model.safetensors").exists():
            continue
        state = json.load(open(state_path, encoding="utf-8"))
        epoch = int(round(float(state["epoch"])))
        found.append((epoch, checkpoint))
    found.sort()
    if [epoch for epoch, _ in found] != list(range(1, 9)):
        raise RuntimeError(f"expected epoch checkpoints 1-8, found {[epoch for epoch, _ in found]}")
    return found


def bootstrap_gap(audit):
    member = np.asarray([row["loss"] for row in audit["exposure"]["member_losses"]])
    reference = np.asarray([row["loss"] for row in audit["exposure"]["reference_losses"]])
    rng = np.random.default_rng(42)
    draws = np.asarray([
        rng.choice(reference, len(reference)).mean() - rng.choice(member, len(member)).mean()
        for _ in range(2000)
    ])
    return {
        "estimate": float(reference.mean() - member.mean()),
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "confidence": 0.95,
        "n_bootstrap": 2000,
        "seed": 42,
    }


def main():
    os.environ.update(
        CUDA_VISIBLE_DEVICES="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    tasks = load_validation()
    validation_dir = OUT / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for epoch, checkpoint in epoch_checkpoints():
        print(f"[validation] epoch {epoch}: {checkpoint}", flush=True)
        metrics, detail = evaluate_epoch(checkpoint, epoch, tasks)
        record_validation(
            MANIFEST, CANDIDATE_ID, epoch=epoch, checkpoint=str(checkpoint), metrics=metrics
        )
        with open(validation_dir / f"epoch_{epoch:02d}.json", "w", encoding="utf-8") as handle:
            json.dump(detail, handle, indent=2)
        details.append(detail)
    with open(validation_dir / "all_epochs.json", "w", encoding="utf-8") as handle:
        json.dump({"evaluated_epochs": list(range(1, 9)), "results": details}, handle, indent=2)

    selected = select_search(MANIFEST)
    epoch = selected["selected"]["epoch"]
    source = Path(selected["selected"]["checkpoint"])
    selected_dir = OUT / f"selected_checkpoint_epoch_{epoch:02d}"
    if selected_dir.exists():
        raise RuntimeError(f"refusing to overwrite selected checkpoint: {selected_dir}")
    shutil.copytree(source, selected_dir)
    selected["selected"]["checkpoint"] = str(selected_dir)
    with open(OUT / "selected.json", "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    with open(OUT / "checkpoint_selection.json", "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    print(f"[audit] selected epoch {epoch}", flush=True)
    artifacts = SimpleNamespace(
        member_path=str(DATA / "member_canaries.jsonl"),
        reference_path=str(DATA / "attack_eval.jsonl"),
    )
    audit = audit_checkpoint(
        str(BASE), str(selected_dir), artifacts, device="cuda",
        max_member_samples=770, max_reference_samples=1000,
        reference_model_path=str(BASE), run_wbc=True, run_extraction=True,
        n_extraction_targets=20, mask_prompt=False,
    ).to_dict()
    audit["base_model"] = "google/gemma-2-2b-it"
    audit["offline_enforced"] = True
    audit["checkpoint_selection"] = selected
    audit["memorization_gap_bootstrap"] = bootstrap_gap(audit)
    with open(OUT / "canarybench_audit_full.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    exposure = audit["exposure"]
    wbc = audit["wbc"]
    summary = {
        "model": "gemma_2b_it",
        "privacy_mode": "none",
        "seed": 42,
        "selected_epoch": epoch,
        "selected_checkpoint": str(selected_dir),
        "validation_metrics": selected["selected"]["validation"],
        "member_loss": exposure["member_loss_mean"],
        "reference_loss": exposure["reference_loss_mean"],
        "memorization_gap": exposure["gap"],
        "memorization_gap_bootstrap": audit["memorization_gap_bootstrap"],
        "gap_by_frequency": audit["frequency_breakdown"],
        "wbc": {key: wbc[key] for key in ("wbc_auc", "auc_ci_lower", "auc_ci_upper", "loss_baseline_auc")},
        "exposure": {key: exposure[key] for key in ("exposure_mean", "exposure_p50", "exposure_p95", "exposure_max")},
        "targeted_greedy_extraction": audit["targeted_extraction"],
        "evaluated_epochs": list(range(1, 9)),
        "test_set_touched": False,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[post] COMPLETE: {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

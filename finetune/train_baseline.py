"""
Baseline LoRA finetuning — no differential privacy.

Usage:
    # Pilot run (debug, ~20 mins)
    python train_baseline.py --model gemma-2-2b-it --pilot

    # Full run
    python train_baseline.py --model gemma-2-2b-it
    python train_baseline.py --model gemma_2b
    python train_baseline.py --model vault_gemma
    python train_baseline.py --model llama
"""

import os
import sys
import time
import json
import argparse
import logging
import shutil
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add the project root to Python path to ensure benchmarks module is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODELS, TrainingConfig, require_dataset_paths
from canary_eval import evaluate_canaries_full_sequence
from dataset import MultiTaskDataset

try:
    from benchmarks.finprivcanary import evaluate_canaries_on_secret
except ImportError:
    evaluate_canaries_on_secret = None


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logger(log_path):
    logger = logging.getLogger("baseline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Per-step loss callback ────────────────────────────────────────────────────

class StepLossCallback(TrainerCallback):
    def __init__(self):
        self.step_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.step_losses.append({
                "step": state.global_step,
                "loss": round(logs["loss"], 4),
                "epoch": round(state.epoch, 3) if state.epoch else 0,
            })


class CustomDataCollator:
    """
    Custom data collator that excludes metadata fields (group, task) from tokenization.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
        )

    def __call__(self, features):
        # Extract metadata fields that shouldn't be tokenized
        metadata = {
            "group": [f.pop("group", None) for f in features],
            "task": [f.pop("task", None) for f in features],
        }
        
        # Process the remaining features with the standard data collator
        batch = self.data_collator(features)
        
        # Add metadata back to the batch
        batch.update(metadata)
        
        return batch


# ── Result saver ──────────────────────────────────────────────────────────────

def save_results(results, results_dir, model_key, run_type):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{model_key}_{run_type}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def save_trainer_state_snapshot(trainer, output_dir):
    """
    Persist trainer state outside rotating checkpoint directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    trainer.state.save_to_json(trainer_state_path)
    return trainer_state_path


def _resolve_pretrained_source(model_name: str):
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path), {"local_files_only": True}
    if path.is_absolute():
        raise FileNotFoundError(
            f"Local model path does not exist: {path}. "
            "Update finetune/config.py to a valid local directory or a Hugging Face model ID."
        )
    return model_name, {}


def _load_tokenizer(model_source: str, hf_token: str, source_kwargs: dict):
    model_name_lower = str(model_source).lower()
    if "gemma" in model_name_lower:
        return GemmaTokenizer.from_pretrained(
            model_source,
            token=hf_token,
            **source_kwargs,
        )

    try:
        return AutoTokenizer.from_pretrained(
            model_source,
            token=hf_token,
            **source_kwargs,
        )
    except Exception as exc:
        error_text = str(exc)
        slow_tokenizer_errors = (
            "SentencePieceExtractor requires the protobuf library",
            "Error parsing line",
            "tiktoken",
            "convert_slow_tokenizer",
        )
        if not any(marker in error_text for marker in slow_tokenizer_errors):
            raise

    try:
        return AutoTokenizer.from_pretrained(
            model_source,
            token=hf_token,
            use_fast=False,
            **source_kwargs,
        )
    except Exception:
        return GemmaTokenizer.from_pretrained(
            model_source,
            token=hf_token,
            **source_kwargs,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def train_baseline(cfg: TrainingConfig, pilot: bool = False):

    # Seed before model loading/LoRA construction so adapter initialization is
    # reproducible, not merely the later Trainer data order.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if pilot:
        cfg.num_epochs    = cfg.pilot_epochs
        cfg.eval_steps    = 50
        cfg.save_steps    = 50
        cfg.logging_steps = 10

    run_label  = "baseline_pilot" if pilot else "baseline"
    output_dir = os.path.join(cfg.output_base, cfg.model_key, run_label)
    log_path   = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info(f"Baseline LoRA {'[PILOT] ' if pilot else ''}— {cfg.model_name}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("=" * 60)

    cuda_available = torch.cuda.is_available()
    if cfg.device == "cuda" and cuda_available:
        device = torch.device("cuda")
        device_map = "auto"
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
    else:
        device = torch.device("cpu")
        device_map = None
        use_bf16 = False
        use_fp16 = False

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    logger.info(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    logger.info(f"Device   : {device}")
    logger.info(f"cfg.device: {cfg.device}")
    start_time = time.time()
    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    model_source, source_kwargs = _resolve_pretrained_source(cfg.model_name)
    tokenizer = _load_tokenizer(model_source, cfg.hf_token, source_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype      = torch.float32 if device.type == "cpu" else torch.float16,
        device_map = device_map,
        token      = cfg.hf_token,
        **source_kwargs,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    logger.info(f"Applying LoRA  r={cfg.lora_r}  alpha={cfg.lora_alpha}")
    lora_config = LoraConfig(
        r              = cfg.lora_r,
        lora_alpha     = cfg.lora_alpha,
        lora_dropout   = cfg.lora_dropout,
        target_modules = cfg.target_modules,
        task_type      = TaskType.CAUSAL_LM,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} "
                f"({100*trainable_params/total_params:.2f}%)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_max = cfg.pilot_train_size if pilot else None
    val_max   = cfg.pilot_val_size   if pilot else None

    train_ds = MultiTaskDataset(cfg.train_path, tokenizer, cfg.max_seq_length,
                                max_records=train_max)
    val_ds   = MultiTaskDataset(cfg.val_path,   tokenizer, cfg.max_seq_length,
                                max_records=val_max)
    logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Training args ─────────────────────────────────────────────────────────
    checkpoint_strategy = "epoch" if cfg.save_epoch_checkpoints else "steps"
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = cfg.num_epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        gradient_accumulation_steps = cfg.grad_accum,
        learning_rate               = cfg.learning_rate,
        warmup_steps                = cfg.warmup_steps,
        eval_strategy               = checkpoint_strategy,
        eval_steps                  = cfg.eval_steps,
        save_strategy               = checkpoint_strategy,
        save_steps                  = cfg.save_steps,
        save_total_limit            = None if cfg.keep_all_checkpoints else 2,
        logging_steps               = cfg.logging_steps,
        seed                        = cfg.seed,
        data_seed                   = cfg.seed,
        fp16                        = (device.type == "cuda" and not torch.cuda.is_bf16_supported()),
        bf16                        = (device.type == "cuda" and torch.cuda.is_bf16_supported()),
        report_to                   = "none",
        load_best_model_at_end      = not cfg.save_epoch_checkpoints,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
    )

    step_callback = StepLossCallback()

    # Use custom data collator to handle metadata fields
    data_collator = CustomDataCollator(tokenizer)

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = data_collator,
        callbacks        = [step_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    train_result = trainer.train()

    # ── Canary evaluation ─────────────────────────────────────────────────────
    # NOTE: the four `secret_*` variables below (secret_canary_results,
    # secret_reference_results, avg_secret_canary, avg_secret_reference) were
    # previously referenced (in the results dict) without ever being
    # defined -- a NameError on any real (non-pilot) run. Restored here by
    # mirroring finetune/train_dp.py's identical guarded-import pattern
    # (evaluate_canaries_on_secret imported with a try/except ImportError
    # fallback to None) rather than inventing new behavior. That function
    # does not actually exist anywhere in this repo -- not in
    # benchmarks.finprivcanary's exports, not in any stale __pycache__ --
    # so in train_dp.py this guarded block is *also* currently a no-op: it
    # never executes, and secret_canary_results/etc. stay at these safe
    # defaults there too. Restoring the same defaults+guard here just makes
    # train_baseline.py behave consistently with train_dp.py (both
    # currently produce empty/0.0 secret-canary metrics) instead of
    # crashing. avg_canary/avg_reference are likewise given safe defaults
    # so pilot runs (which skip this block) don't NameError either.
    canary_results    = []
    reference_results = []
    avg_canary        = 0.0
    avg_reference     = 0.0
    secret_canary_results    = []
    secret_reference_results = []
    avg_secret_canary        = 0.0
    avg_secret_reference     = 0.0
    if cfg.eval_canaries and not pilot:
        logger.info("Evaluating canary losses...")

        canary_results, avg_canary = evaluate_canaries_full_sequence(
            model, tokenizer, cfg.canary_path, device, cfg.canary_eval_samples
        )
        reference_results, avg_reference = evaluate_canaries_full_sequence(
            model, tokenizer, cfg.reference_path, device, cfg.canary_eval_samples
        )
        logger.info(f"  Avg member loss    : {avg_canary:.4f}")
        logger.info(f"  Avg reference loss : {avg_reference:.4f}")
        logger.info(f"  Gap (ref - member) : {avg_reference - avg_canary:.4f}")
        logger.info("  (larger gap = more memorization of member canaries)")

        if (
            evaluate_canaries_on_secret is not None
            and os.path.exists(cfg.canary_registry_path)
        ):
            secret_canary_results, avg_secret_canary, _ = evaluate_canaries_on_secret(
                model,
                tokenizer,
                cfg.canary_path,
                cfg.canary_registry_path,
                device,
                cfg.canary_eval_samples,
            )
            secret_reference_results, avg_secret_reference, _ = evaluate_canaries_on_secret(
                model,
                tokenizer,
                cfg.reference_path,
                cfg.canary_registry_path,
                device,
                cfg.canary_eval_samples,
            )
    else:
        logger.info("Skipping canary eval in pilot mode")

    # ── Save model ────────────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer_state_path = save_trainer_state_snapshot(trainer, final_dir)
    logger.info(f"Model saved → {final_dir}")
    logger.info(f"Trainer state saved → {trainer_state_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results = {
        "model_key":         cfg.model_key,
        "model_name":        cfg.model_name,
        "run_type":          "baseline",
        "seed":              cfg.seed,
        "pilot":             pilot,
        "train_loss":        round(train_result.training_loss, 4),
        "train_runtime_sec": round(elapsed, 2),
        "train_runtime_hrs": round(elapsed / 3600, 3),
        "train_samples":     len(train_ds),
        "val_samples":       len(val_ds),
        "epochs":            cfg.num_epochs,
        "batch_size":        cfg.batch_size,
        "grad_accum":        cfg.grad_accum,
        "effective_batch":   cfg.batch_size * cfg.grad_accum,
        "learning_rate":     cfg.learning_rate,
        "lora_r":            cfg.lora_r,
        "lora_alpha":        cfg.lora_alpha,
        "target_modules":    cfg.target_modules,
        "trainable_params":  trainable_params,
        "total_params":      total_params,
        "step_losses":       step_callback.step_losses,
        "canary_losses":     canary_results,
        "member_loss_mean":   round(avg_canary, 4),
        "reference_loss_mean": round(avg_reference, 4),
        "canary_gap":         round(avg_reference - avg_canary, 4),
        "secret_canary_losses": secret_canary_results,
        "secret_reference_losses": secret_reference_results,
        "member_secret_loss_mean": round(avg_secret_canary, 4),
        "reference_secret_loss_mean": round(avg_secret_reference, 4),
        "secret_canary_gap": round(avg_secret_reference - avg_secret_canary, 4),
        "epsilon":            "inf",
        "noise_multiplier":   0.0,
        "peak_gpu_mem_gb":    round(torch.cuda.max_memory_allocated()/1e9, 2),
        "reference_losses":  reference_results,
        "trainer_log_history": trainer.state.log_history,
        "trainer_state_path": trainer_state_path,
        "output_dir":        final_dir,
        "timestamp":         datetime.now().isoformat(),
    }

    # Pilot checklist
    if pilot:
        loss_ok  = 0 < train_result.training_loss < 10.0
        saved_ok = os.path.exists(os.path.join(final_dir, "adapter_config.json"))
        logger.info("\n" + "=" * 50)
        logger.info("PILOT CHECKLIST")
        logger.info("=" * 50)
        logger.info(f"  Loss reasonable : {'✅' if loss_ok  else '❌'}  ({train_result.training_loss:.4f})")
        logger.info(f"  Model saved     : {'✅' if saved_ok else '❌'}")
        logger.info(f"  Runtime         : {elapsed/60:.1f} mins")
        status = "✅ PASSED — safe to run full training" if (loss_ok and saved_ok) else "❌ FAILED — fix before full run"
        logger.info(f"\n  {status}")
        logger.info("=" * 50)

    results_path = save_results(results, cfg.results_dir, cfg.model_key, run_label)
    stable_state_path = os.path.splitext(results_path)[0] + "_trainer_state.json"
    shutil.copy2(trainer_state_path, stable_state_path)
    logger.info(f"Results → {results_path}")
    logger.info(f"Trainer state snapshot → {stable_state_path}")
    logger.info(f"Time    : {elapsed/3600:.2f} hrs")
    logger.info("✅ Done")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str,  required=True, choices=list(MODELS.keys()))
    parser.add_argument("--pilot",     action="store_true", help="Quick debug run (~20 mins)")
    parser.add_argument("--hf_token",  type=str,  default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--canary-path", type=str, default=None)
    parser.add_argument("--reference-path", type=str, default=None)
    parser.add_argument("--canary-registry-path", type=str, default=None)
    parser.add_argument("--output-base", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = replace(MODELS[args.model])
    if args.hf_token:
        cfg.hf_token = args.hf_token
    if args.train_path:
        cfg.train_path = args.train_path
    if args.val_path:
        cfg.val_path = args.val_path
    if args.canary_path:
        cfg.canary_path = args.canary_path
    if args.reference_path:
        cfg.reference_path = args.reference_path
    if args.canary_registry_path:
        cfg.canary_registry_path = args.canary_registry_path
    if args.output_base:
        cfg.output_base = args.output_base
    if args.results_dir:
        cfg.results_dir = args.results_dir

    require_dataset_paths(cfg)
    train_baseline(cfg, pilot=args.pilot)

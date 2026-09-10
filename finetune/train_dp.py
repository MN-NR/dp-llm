"""
DP-LoRA finetuning with Opacus differential privacy.

Usage:
    # Pilot run first
    python train_dp.py --model gemma-2-2b-it --epsilon 8.0 --pilot

    # Full runs
    python train_dp.py --model gemma-2-2b-it --epsilon 8.0
    python train_dp.py --model gemma-2-2b-it --epsilon 5.0
    python train_dp.py --model gemma-2-2b-it --epsilon 3.0
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import json
import random
import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

# Add the project root to Python path to ensure benchmarks module is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import MODELS, TrainingConfig, require_dataset_paths
from canary_eval import evaluate_canaries_full_sequence
from dataset import MultiTaskDataset
from benchmarks.finprivcanary import (
    load_canaries_for_checkpointing,
    CrashSafeCallback,
)

try:
    from benchmarks.finprivcanary import evaluate_canaries_on_secret
except ImportError:
    evaluate_canaries_on_secret = None


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path):
    logger = logging.getLogger("dp_lora")
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


# ── Crash-safe checkpoint callback ────────────────────────────────────────────
# Note: CrashSafeCallback is now imported from benchmarks.finprivcanary



# ── Validation ────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device):
    model.eval()
    total, steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device),
            )
            total += out.loss.item()
            steps += 1
    model.train()
    return total / max(steps, 1)


# ── Result saver ──────────────────────────────────────────────────────────────

def save_results(results, results_dir, model_key, epsilon, seed):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag   = f"dp_eps{epsilon}_seed{seed}"
    path = os.path.join(results_dir, f"{model_key}_{run_tag}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def _named_parameters(module):
    if hasattr(module, "_module"):
        return module._module.named_parameters()
    return module.named_parameters()


def _compute_norm(params):
    total = 0.0
    for p in params:
        if p is not None:
            total += p.data.norm(2).item() ** 2
    return total ** 0.5


def _compute_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _get_gpu_memory_stats(device):
    """Get GPU memory usage in GB."""
    if "cuda" not in str(device):
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return allocated, reserved


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

def train_dp(
    cfg: TrainingConfig,
    epsilon: float,
    pilot: bool = False,
    seed: int = 42,
    noise_multiplier: "float | None" = None,
):
    """noise_multiplier: leave None for the normal DP path (Opacus
    make_private_with_epsilon() calibrates noise to hit `epsilon` over
    cfg.num_epochs -- unchanged from before this parameter existed). Pass an
    explicit value (e.g. 0.0) to skip epsilon calibration entirely and call
    Opacus make_private() directly with that noise_multiplier and
    cfg.max_grad_norm -- e.g. noise_multiplier=0.0 gives a "clipping-only"
    mechanistic ablation (per-sample gradient clipping applied, no DP noise
    added) to isolate clipping's contribution from noise's. Opacus's own
    get_epsilon() already returns float("inf") (not an exception) when
    noise_multiplier=0, so no special-casing is needed anywhere else in this
    function -- every existing epsilon log/report line already handles it.
    """

    # if pilot:
    #     cfg.num_epochs    = cfg.pilot_epochs
    #     cfg.logging_steps = 10
    if pilot:
        cfg.num_epochs         = cfg.pilot_epochs
        cfg.pilot_train_size   = None   # use FULL dataset for DP accounting
        cfg.pilot_val_size     = 250    # keep val small
        cfg.num_epochs         = 1      # just 1 epoch

    if noise_multiplier is not None:
        run_tag = f"clip_only_C{cfg.max_grad_norm}_seed{seed}"
    else:
        run_tag = f"dp_eps{epsilon}_seed{seed}"
    if pilot:
        run_tag += "_pilot"
    output_dir = os.path.join(cfg.output_base, cfg.model_key, run_tag)
    log_path   = os.path.join(output_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(log_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if pilot:
        logger.warning("PILOT: using full dataset for correct DP accounting")
    logger.info("=" * 60)
    logger.info(f"DP-LoRA {'[PILOT] ' if pilot else ''}— {cfg.model_name}  target ε={epsilon}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Seed       : {seed}")
    logger.info("=" * 60)

    cuda_available = torch.cuda.is_available()
    if cfg.device == "cuda" and cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    logger.info(f"Device    : {device}")
    logger.info(f"cfg.device: {cfg.device}")
    start_time = time.time()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    model_source, source_kwargs = _resolve_pretrained_source(cfg.model_name)
    tokenizer = _load_tokenizer(model_source, cfg.hf_token, source_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model — bfloat16 for DP stability ───────────────────────────────────
    logger.info("Loading model (bfloat16 for DP stability)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
        **source_kwargs,
    )

    # Fix Opacus-incompatible layers
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info(f"Fixing {len(errors)} Opacus-incompatible layers...")
        model = ModuleValidator.fix(model)

    torch.cuda.empty_cache()
    model = model.to(device)

    # ── LoRA — dropout=0 required for Opacus ──────────────────────────────────
    logger.info(f"Applying LoRA  r={cfg.lora_r}  alpha={cfg.lora_alpha}  dropout=0.0")
    lora_config = LoraConfig(
        r              = cfg.lora_r,
        lora_alpha     = cfg.lora_alpha,
        lora_dropout   = 0.0,
        target_modules = cfg.target_modules,
        task_type      = TaskType.CAUSAL_LM,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    # Transformers checkpoint recomputation prevents Opacus 1.5.4 hooks from
    # producing grad_sample for every LoRA parameter. Physical microbatching is
    # the supported memory-control mechanism on this DP path.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for Opacus compatibility")
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

    # ── Load canaries for crash-safe checkpointing ────────────────────────────
    member_canaries = []
    reference_canaries = []
    
    if cfg.eval_canaries and not pilot:
        logger.info("Loading canaries for crash-safe checkpointing...")
        member_canaries, reference_canaries = load_canaries_for_checkpointing(
            cfg.canary_path, cfg.reference_path, tokenizer, cfg.canary_eval_samples
        )

    def collate_fn(batch):
        if len(batch) == 0:
            return {}
        result = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            if isinstance(tensors[0], torch.Tensor):
                result[key] = torch.stack(tensors)
            else:
                result[key] = tensors
        return result

    logical_batch_size = cfg.batch_size * cfg.grad_accum
    if cfg.dp_physical_batch_size > logical_batch_size:
        raise ValueError(
            "dp_physical_batch_size must not exceed the logical DP batch size "
            f"({cfg.dp_physical_batch_size} > {logical_batch_size})"
        )
    logger.info(
        f"DP batch sizes: logical={logical_batch_size}, "
        f"physical<={cfg.dp_physical_batch_size}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = logical_batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        collate_fn  = collate_fn,
        generator   = torch.Generator().manual_seed(seed),
        # NO sampler= line at all
    )
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Debug: Check what the batch actually contains
    print("Debug: Checking batch contents...")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}, value={str(v)[:50]}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = cfg.learning_rate,
    )

    # ── Privacy engine ────────────────────────────────────────────────────────
    if noise_multiplier is not None:
        logger.info(f"Attaching PrivacyEngine  CLIPPING-ONLY  noise_multiplier={noise_multiplier}  "
                    f"clip={cfg.max_grad_norm}")
    else:
        logger.info(f"Attaching PrivacyEngine  ε={epsilon}  δ={cfg.delta}  "
                    f"clip={cfg.max_grad_norm}")
    # Force use of analytical accountant to avoid PRV bug
    import opacus.accountants.rdp as rdp_module
    rdp_module.PRV_AVAILABLE = False
    privacy_engine = PrivacyEngine(accountant='rdp')
    noise_generator = torch.Generator(device=device.type).manual_seed(seed)

    print(f"max_grad_norm being used: {cfg.max_grad_norm}")

    if noise_multiplier is not None:
        # Clipping-only mechanistic ablation: per-sample gradient clipping
        # to cfg.max_grad_norm, explicit noise_multiplier (0.0 = no DP noise
        # added at all), no epsilon calibration/target.
        model, optimizer, train_loader = privacy_engine.make_private(
            module          = model,
            optimizer       = optimizer,
            data_loader     = train_loader,
            noise_multiplier = noise_multiplier,
            max_grad_norm   = cfg.max_grad_norm,
            noise_generator = noise_generator,
        )
    else:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module         = model,
            optimizer      = optimizer,
            data_loader    = train_loader,
            target_epsilon = epsilon,
            target_delta   = cfg.delta,
            epochs         = cfg.num_epochs,
            max_grad_norm  = cfg.max_grad_norm,
            noise_generator = noise_generator,
        )

    # Add noise multiplier and sample rate prints
    print(f"Noise multiplier: {optimizer.noise_multiplier:.4f}")
    print(f"Sample rate: {optimizer.expected_batch_size / len(train_loader.dataset):.8f}")

    # ── Fix Opacus collate_fn bug ─────────────────────────────────────────────
    import opacus.data_loader as odl

    def _fixed_collate(batch):
        if len(batch) == 0:
            return {}
        result = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            if isinstance(tensors[0], torch.Tensor):
                result[key] = torch.stack(tensors)
            else:
                result[key] = tensors
        return result

    train_loader.collate_fn = _fixed_collate

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = cfg.warmup_steps,
        num_training_steps = total_steps,
    )

    print(f"Sample rate: {logical_batch_size / len(train_ds):.8f}")
    # ── Crash-safe callback ───────────────────────────────────────────────────
    crash_callback = None
    if not pilot:  # Only use crash-safe for full runs
        crash_callback = CrashSafeCallback(
            privacy_engine=privacy_engine,

            delta=cfg.delta,
            member_canaries=member_canaries,
            reference_canaries=reference_canaries,
            save_every_steps=500
        )
        logger.info("✅ Crash-safe checkpoint system enabled")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epoch_results = []
    all_step_losses = []
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss, steps = 0.0, 0

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=cfg.dp_physical_batch_size,
            optimizer=optimizer,
        ) as safe_loader:
            for physical_step, batch in enumerate(safe_loader):
                if len(batch) == 0:
                    continue
                optimizer.zero_grad()
    
                outputs = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    labels         = batch["labels"].to(device),
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logical_step_complete = not optimizer._is_last_step_skipped
                if logical_step_complete:
                    scheduler.step()
                    global_step += 1
    
                loss_val = loss.item()
                del outputs, loss
                if physical_step % 50 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
    
                # Checkpoint and log only after a complete logical DP step.
                if (
                    logical_step_complete
                    and crash_callback
                    and global_step % 500 == 0
                ):
                    class State:
                        def __init__(self, step, epoch):
                            self.global_step = step
                            self.epoch = epoch
                    crash_callback.on_step_end(State(global_step, epoch + 1), model)
    
                epoch_loss += loss_val
                steps      += 1
    
                if (
                    logical_step_complete
                    and global_step % cfg.logging_steps == 0
                ):
                    eps_now = privacy_engine.get_epsilon(cfg.delta)
    
                    total_norm = _compute_grad_norm(model.parameters())
                    lora_grad_norm = _compute_grad_norm(
                        p for name, p in _named_parameters(model)
                        if "lora_B" in name
                    )
                    lora_weight_norm = _compute_norm(
                        p for name, p in _named_parameters(model)
                        if "lora_B" in name
                    )
                    
                    gpu_allocated, gpu_reserved = _get_gpu_memory_stats(device)
    
                    logger.info(
                        f"Epoch {epoch+1}/{cfg.num_epochs}  "
                        f"Step {global_step:>6}  "
                        f"loss: {loss_val:.4f}  "
                        f"grad_norm: {total_norm:.4f}  "
                        f"lora_weight_norm: {lora_weight_norm:.4f}  "
                        f"lora_grad_norm: {lora_grad_norm:.4f}  "
                        f"noise_multiplier: {optimizer.noise_multiplier:.4f}  "
                        f"ε: {eps_now:.3f}  "
                        f"gpu_mem: {gpu_allocated:.2f}GB / {gpu_reserved:.2f}GB"
                    )
                    all_step_losses.append({
                        "epoch": epoch + 1,
                        "step":  global_step,
                        "loss":  round(loss_val, 4),
                        "epsilon_spent": round(eps_now, 4),
                        "grad_norm": round(total_norm, 6),
                        "lora_weight_norm": round(lora_weight_norm, 6),
                        "lora_grad_norm": round(lora_grad_norm, 6),
                        "noise_multiplier": round(optimizer.noise_multiplier, 6),
                        "gpu_allocated_gb": round(gpu_allocated, 2),
                        "gpu_reserved_gb": round(gpu_reserved, 2),
                    })


        # ── End of epoch ──────────────────────────────────────────────────────
        avg_train = epoch_loss / max(steps, 1)
        val_loss  = evaluate(model, val_loader, device)
        eps_spent = privacy_engine.get_epsilon(cfg.delta)

        # ── Canary losses this epoch — record every epoch ──────────────────
        canary_results    = []
        reference_results = []
        avg_canary        = 0.0
        avg_reference     = 0.0
        secret_canary_results = []
        secret_reference_results = []
        avg_secret_canary = 0.0
        avg_secret_reference = 0.0

        if cfg.eval_canaries and not pilot:
            canary_results, avg_canary = evaluate_canaries_full_sequence(
                model, tokenizer, cfg.canary_path, device, cfg.canary_eval_samples
            )
            reference_results, avg_reference = evaluate_canaries_full_sequence(
                model, tokenizer, cfg.reference_path, device, cfg.canary_eval_samples
            )
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

        logger.info(f"\n{'─'*55}")
        logger.info(f"Epoch {epoch+1} Summary")
        logger.info(f"  Train loss         : {avg_train:.4f}")
        logger.info(f"  Val loss           : {val_loss:.4f}")
        logger.info(f"  ε spent            : {eps_spent:.4f}  (target: {epsilon})")
        if cfg.eval_canaries and not pilot:
            logger.info(f"  Avg member loss    : {avg_canary:.4f}")
            logger.info(f"  Avg reference loss : {avg_reference:.4f}")
            logger.info(f"  Gap (ref - member) : {avg_reference - avg_canary:.4f}")
            if secret_canary_results and secret_reference_results:
                logger.info(f"  Avg member secret loss    : {avg_secret_canary:.4f}")
                logger.info(f"  Avg reference secret loss : {avg_secret_reference:.4f}")
                logger.info(f"  Secret gap (ref - member) : {avg_secret_reference - avg_secret_canary:.4f}")
        logger.info(f"{'─'*55}\n")

        epoch_results.append({
            "epoch":             epoch + 1,
            "train_loss":        round(avg_train, 4),
            "val_loss":          round(val_loss, 4),
            "epsilon_spent":     round(eps_spent, 4),
            "avg_canary_loss":   round(avg_canary, 4),
            "avg_reference_loss":round(avg_reference, 4),
            "avg_secret_canary_loss": round(avg_secret_canary, 4),
            "avg_secret_reference_loss": round(avg_secret_reference, 4),
            "canary_losses":     canary_results,
            "reference_losses":  reference_results,
            "secret_canary_losses": secret_canary_results,
            "secret_reference_losses": secret_reference_results,
        })

        # Tuning runs need every epoch available for validation-utility
        # selection. Checkpoint creation remains a trainer responsibility;
        # CanaryBench only opts into this existing-trainer behavior.
        if cfg.save_epoch_checkpoints:
            epoch_dir = os.path.join(output_dir, "epoch_checkpoints", f"epoch_{epoch + 1:02d}")
            model._module.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            with open(os.path.join(epoch_dir, "training_metadata.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "optimizer_steps": global_step,
                    "epsilon_actual": eps_spent,
                    "epsilon_target": epsilon,
                    "delta": cfg.delta,
                    "noise_multiplier": optimizer.noise_multiplier,
                    "accountant": "rdp",
                }, f, indent=2)
            logger.info(f"  Epoch checkpoint saved → {epoch_dir}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(output_dir, "best")
            model._module.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"  ✅ Best model saved  val_loss={val_loss:.4f}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir     = os.path.join(output_dir, "final")
    final_epsilon = privacy_engine.get_epsilon(cfg.delta)
    model._module.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved → {final_dir}")
    logger.info(f"Final ε = {final_epsilon:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results = {
        "model_key":          cfg.model_key,
        "model_name":         cfg.model_name,
        "run_type":           "dp_lora",
        "pilot":              pilot,
        "seed":               seed,
        "clipping_only":      noise_multiplier is not None,
        "target_epsilon":     epsilon,
        "final_epsilon":      round(final_epsilon, 4),
        "delta":              cfg.delta,
        "max_grad_norm":      cfg.max_grad_norm,
        "noise_multiplier":   round(optimizer.noise_multiplier, 4),
        "best_val_loss":      round(best_val_loss, 4),
        "peak_gpu_mem_gb":    round(torch.cuda.max_memory_allocated()/1e9, 2),
        "train_runtime_sec":  round(elapsed, 2),
        "train_runtime_hrs":  round(elapsed / 3600, 3),
        "train_samples":      len(train_ds),
        "val_samples":        len(val_ds),
        "epochs":             cfg.num_epochs,
        "batch_size":         cfg.batch_size,
        "grad_accum":         cfg.grad_accum,
        "logical_batch_size": logical_batch_size,
        "dp_physical_batch_size": cfg.dp_physical_batch_size,
        "sample_rate": logical_batch_size / len(train_ds),
        "optimizer_steps": global_step,
        "accountant": "rdp",
        "learning_rate":      cfg.learning_rate,
        "lora_r":             cfg.lora_r,
        "lora_alpha":         cfg.lora_alpha,
        "target_modules":     cfg.target_modules,
        "trainable_params":   trainable_params,
        "total_params":       total_params,
        "step_losses":        all_step_losses,
        "epoch_results":      epoch_results,
        "output_dir":         final_dir,
        "timestamp":          datetime.now().isoformat(),
    }

    # Pilot checklist
    if pilot:
        loss_ok  = 0 < epoch_results[-1]["train_loss"] < 10.0
        saved_ok = os.path.exists(os.path.join(final_dir, "adapter_config.json"))
        eps_ok   = epoch_results[-1]["epsilon_spent"] <= epsilon * 1.1
        logger.info("\n" + "=" * 50)
        logger.info("PILOT CHECKLIST")
        logger.info("=" * 50)
        logger.info(f"  Loss reasonable : {'✅' if loss_ok  else '❌'}  ({epoch_results[-1]['train_loss']:.4f})")
        logger.info(f"  Model saved     : {'✅' if saved_ok else '❌'}")
        logger.info(f"  ε within budget : {'✅' if eps_ok   else '❌'}  ({epoch_results[-1]['epsilon_spent']:.4f} vs target {epsilon})")
        logger.info(f"  Runtime         : {elapsed/60:.1f} mins")
        status = "✅ PASSED" if (loss_ok and saved_ok and eps_ok) else "❌ FAILED"
        logger.info(f"\n  {status}")
        logger.info("=" * 50)

    results_path = save_results(
        results, cfg.results_dir, cfg.model_key, epsilon, seed
    )
    logger.info(f"Results → {results_path}")
    logger.info(f"Time    : {elapsed/3600:.2f} hrs")
    logger.info("✅ Done")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str,   required=True, choices=list(MODELS.keys()))
    parser.add_argument("--epsilon",  type=float, required=True, help="Target ε (3.0, 5.0, 8.0)")
    parser.add_argument("--seed",     type=int,   default=42, help="Random seed (default: 42)")
    parser.add_argument("--pilot",    action="store_true", help="Quick debug run")
    parser.add_argument("--hf_token", type=str,   default=None)
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
    train_dp(cfg, epsilon=args.epsilon, pilot=args.pilot, seed=args.seed)

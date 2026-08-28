"""
Two-GPU DPDDP finetuning for LS04 (2x L40S, 46GB each).

Same DP-LoRA pipeline as train_dp.py (canary eval, crash-safe checkpointing,
grad-norm logging), but wraps the model in Opacus's
DifferentiallyPrivateDistributedDataParallel so the Poisson-sampled logical
batch is split across both GPUs. That halves wall-clock per epoch at a fixed
logical batch size, or -- since config.py already targets 2x L40S with plenty
of headroom -- lets you raise --batch-size/--grad-accum well past the
single-GPU dp_physical_batch_size ceiling to improve DP utility at fixed
epsilon (larger batches shrink the per-step noise-to-signal ratio).

Usage (torchrun, exactly 2 processes):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune/train_dp_ddp.py \\
        --model vault_gemma --epsilon 8.0

    # bigger logical batch than config.py's default, split across both GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune/train_dp_ddp.py \\
        --model vault_gemma --epsilon 8.0 --batch-size 8 --grad-accum 16
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
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.elastic.multiprocessing.errors import record
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
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
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


# ── Distributed helpers ──────────────────────────────────────────────────────

def _rank_zero() -> bool:
    return dist.get_rank() == 0


def _base_module(model):
    """Unwrap Opacus's GradSampleModule and DPDDP down to the bare PEFT model."""
    while hasattr(model, "_module") or hasattr(model, "module"):
        model = getattr(model, "_module", getattr(model, "module", model))
    return model


class _RecoveryModelView:
    """Shim so CrashSafeCallback's hardcoded `model._module.save_pretrained(...)`
    (written for the single-GPU GradSampleModule wrap) reaches the bare PEFT
    model instead of the DPDDP wrapper it would otherwise land on."""

    def __init__(self, bare_module):
        self._module = bare_module


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

def train_dp_ddp(
    cfg: TrainingConfig,
    epsilon: float,
    pilot: bool = False,
    seed: int = 42,
):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if pilot:
        cfg.num_epochs         = cfg.pilot_epochs
        cfg.pilot_train_size   = None   # use FULL dataset for DP accounting
        cfg.pilot_val_size     = 250    # keep val small
        cfg.num_epochs         = 1      # just 1 epoch

    # Batch size AND epoch count are both included so distinct experiments
    # never collide on the same output_dir. Batch size was added first (a
    # batch256 run would otherwise silently overwrite a batch128 checkpoint
    # at the same epsilon); epoch count was added after an epochs=12 run
    # silently overwrote an epochs=8 checkpoint's weights at this same path
    # (caught via file mtimes, not before it happened -- see
    # EXPERIMENT_LOG.md's epoch-horizon-sweep section). cfg.num_epochs here
    # already reflects the pilot override above when pilot=True.
    global_batch_size = cfg.batch_size * cfg.grad_accum
    run_tag    = f"dp_eps{epsilon}_seed{seed}_batch{global_batch_size}_epochs{cfg.num_epochs}_2gpu"
    if pilot:
        run_tag += "_pilot"
    output_dir = os.path.join(cfg.output_base, cfg.model_key, run_tag)
    log_path   = os.path.join(output_dir, "train.log")

    if _rank_zero():
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger(log_path)
    else:
        logger = logging.getLogger("dp_lora")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)
    dist.barrier()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if _rank_zero():
        if pilot:
            logger.warning("PILOT: using full dataset for correct DP accounting")
        logger.info("=" * 60)
        logger.info(f"DP-LoRA (2-GPU DPDDP) {'[PILOT] ' if pilot else ''}— {cfg.model_name}  target ε={epsilon}")
        logger.info(f"Output dir : {output_dir}")
        logger.info(f"Seed       : {seed}")
        logger.info(f"World size : {dist.get_world_size()}")
        logger.info("=" * 60)
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"Device     : {device}")

    start_time = time.time()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    model_source, source_kwargs = _resolve_pretrained_source(cfg.model_name)
    tokenizer = _load_tokenizer(model_source, cfg.hf_token, source_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model — bfloat16 for DP stability ───────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
        **source_kwargs,
    )

    # Fix Opacus-incompatible layers
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        if _rank_zero():
            logger.info(f"Fixing {len(errors)} Opacus-incompatible layers...")
        model = ModuleValidator.fix(model)

    torch.cuda.empty_cache()
    model = model.to(device)

    # ── LoRA — dropout=0 required for Opacus ──────────────────────────────────
    if _rank_zero():
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
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        if _rank_zero():
            logger.info("Disabled gradient checkpointing for Opacus compatibility")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    if _rank_zero():
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} "
                    f"({100*trainable_params/total_params:.2f}%)")

    # DPDDP broadcasts rank 0's weights to all ranks and provides the collective
    # gradient sync Opacus's optimizer uses; must wrap after .to(device), before
    # the optimizer is built.
    model = DPDDP(model)

    # ── Datasets ──────────────────────────────────────────────────────────────
    if _rank_zero():
        logger.info("Loading datasets...")
    train_max = cfg.pilot_train_size if pilot else None
    val_max   = cfg.pilot_val_size   if pilot else None

    train_ds = MultiTaskDataset(cfg.train_path, tokenizer, cfg.max_seq_length,
                                max_records=train_max)
    val_ds   = MultiTaskDataset(cfg.val_path,   tokenizer, cfg.max_seq_length,
                                max_records=val_max)
    if _rank_zero():
        logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Load canaries for crash-safe checkpointing (rank 0 only) ──────────────
    member_canaries = []
    reference_canaries = []

    if cfg.eval_canaries and not pilot and _rank_zero():
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

    # This is the GLOBAL logical batch: Opacus detects the DPDDP wrap and
    # splits each Poisson-sampled logical batch across both ranks, so raising
    # batch_size/grad_accum here (via --batch-size/--grad-accum) is what
    # actually buys you a bigger DP batch on LS04's 2 GPUs.
    logical_batch_size = cfg.batch_size * cfg.grad_accum
    if cfg.dp_physical_batch_size > logical_batch_size:
        raise ValueError(
            "dp_physical_batch_size must not exceed the logical DP batch size "
            f"({cfg.dp_physical_batch_size} > {logical_batch_size})"
        )
    if _rank_zero():
        logger.info(
            f"DP batch sizes: logical(global)={logical_batch_size}, "
            f"physical<={cfg.dp_physical_batch_size} per GPU, world_size={dist.get_world_size()}"
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = cfg.learning_rate,
    )

    # ── Privacy engine ────────────────────────────────────────────────────────
    if _rank_zero():
        logger.info(f"Attaching PrivacyEngine  ε={epsilon}  δ={cfg.delta}  "
                    f"clip={cfg.max_grad_norm}")
    # Force use of analytical accountant to avoid PRV bug
    import opacus.accountants.rdp as rdp_module
    rdp_module.PRV_AVAILABLE = False
    privacy_engine = PrivacyEngine(accountant='rdp')
    noise_generator = torch.Generator(device=device.type).manual_seed(seed)

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

    if _rank_zero():
        logger.info(f"Noise multiplier: {optimizer.noise_multiplier:.4f}")
        logger.info(f"Expected batch size (per rank, post-split): {optimizer.expected_batch_size}")
        logger.info(f"Sample rate: {optimizer.expected_batch_size / len(train_loader.dataset):.8f}")

    # ── Fix Opacus collate_fn bug ─────────────────────────────────────────────
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

    # ── Crash-safe callback (rank 0 only) ──────────────────────────────────────
    crash_callback = None
    if not pilot and _rank_zero():
        crash_callback = CrashSafeCallback(
            privacy_engine=privacy_engine,
            delta=cfg.delta,
            member_canaries=member_canaries,
            reference_canaries=reference_canaries,
            save_every_steps=500
        )
        logger.info("✅ Crash-safe checkpoint system enabled")
    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epoch_results = []
    all_step_losses = []
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss_count = torch.zeros(2, device=device)  # summed loss, step count

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

                # Checkpoint and log only after a complete logical DP step,
                # on rank 0 only.
                if (
                    logical_step_complete
                    and crash_callback
                    and global_step % 500 == 0
                ):
                    class State:
                        def __init__(self, step, epoch):
                            self.global_step = step
                            self.epoch = epoch
                    crash_callback.on_step_end(
                        State(global_step, epoch + 1),
                        _RecoveryModelView(_base_module(model)),
                    )

                epoch_loss_count[0] += loss_val
                epoch_loss_count[1] += 1

                if (
                    logical_step_complete
                    and global_step % cfg.logging_steps == 0
                    and _rank_zero()
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

        # ── End of epoch: aggregate train loss across ranks ─────────────────
        dist.all_reduce(epoch_loss_count, op=dist.ReduceOp.SUM)
        avg_train = (epoch_loss_count[0] / epoch_loss_count[1].clamp(min=1)).item()

        # Validation, canary eval, and checkpointing run on the bare (unwrapped)
        # model on rank 0 only -- no need to pay the DDP/GradSampleModule
        # overhead for a forward-only pass, and only one rank should write files.
        if _rank_zero():
            bare_model = _base_module(model)
            val_loss  = evaluate(bare_model, val_loader, device)
            eps_spent = privacy_engine.get_epsilon(cfg.delta)

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
                    bare_model, tokenizer, cfg.canary_path, device, cfg.canary_eval_samples
                )
                reference_results, avg_reference = evaluate_canaries_full_sequence(
                    bare_model, tokenizer, cfg.reference_path, device, cfg.canary_eval_samples
                )
                if (
                    evaluate_canaries_on_secret is not None
                    and os.path.exists(cfg.canary_registry_path)
                ):
                    secret_canary_results, avg_secret_canary, _ = evaluate_canaries_on_secret(
                        bare_model,
                        tokenizer,
                        cfg.canary_path,
                        cfg.canary_registry_path,
                        device,
                        cfg.canary_eval_samples,
                    )
                    secret_reference_results, avg_secret_reference, _ = evaluate_canaries_on_secret(
                        bare_model,
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

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = os.path.join(output_dir, "best")
                bare_model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  ✅ Best model saved  val_loss={val_loss:.4f}")
        dist.barrier()

    # ── Final save (rank 0 only) ────────────────────────────────────────────────
    if _rank_zero():
        bare_model = _base_module(model)
        final_dir     = os.path.join(output_dir, "final")
        final_epsilon = privacy_engine.get_epsilon(cfg.delta)
        bare_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved → {final_dir}")
        logger.info(f"Final ε = {final_epsilon:.4f}")

        # ── Save results ──────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        results = {
            "model_key":          cfg.model_key,
            "model_name":         cfg.model_name,
            "run_type":           "dp_lora_2gpu_ddp",
            "pilot":              pilot,
            "seed":               seed,
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
            "world_size":         dist.get_world_size(),
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
    dist.barrier()


# ── Entry point ───────────────────────────────────────────────────────────────

@record
def main():
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
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config's per-GPU-equivalent batch_size (global logical "
                             "batch = batch_size * grad_accum, split across both GPUs by DPDDP)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override config's grad_accum")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override config's learning_rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config's num_epochs")
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError(
            "train_dp_ddp.py requires exactly two torchrun processes, e.g.:\n"
            "  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune/train_dp_ddp.py ..."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "Expose exactly two CUDA devices, e.g. CUDA_VISIBLE_DEVICES=0,1."
        )

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    try:
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
        if args.batch_size:
            cfg.batch_size = args.batch_size
        if args.grad_accum:
            cfg.grad_accum = args.grad_accum
        if args.learning_rate:
            cfg.learning_rate = args.learning_rate
        if args.epochs:
            cfg.num_epochs = args.epochs

        require_dataset_paths(cfg)
        train_dp_ddp(cfg, epsilon=args.epsilon, pilot=args.pilot, seed=args.seed)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

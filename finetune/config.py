import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))

@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────
    model_name:        str   = "google/gemma-2-2b"
    model_key:         str   = "gemma_2b"
    hf_token:          str   = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", "")
    )

    # ── Device ─────────────────────────────────────────────────
    device:            str   = "cuda"

    # ── LoRA ───────────────────────────────────────────────────
    lora_r:            int   = 16
    lora_alpha:        int   = 32
    lora_dropout:      float = 0.05
    target_modules:    List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # ── Training ───────────────────────────────────────────────
    batch_size:        int   = 4
    grad_accum:        int   = 16
    learning_rate:     float = 1e-5
    num_epochs:        int   = 3
    warmup_steps:      int   = 100
    max_seq_length:    int   = 256
    eval_steps:        int   = 500
    save_steps:        int   = 500
    logging_steps:     int   = 100

    # ── DP ─────────────────────────────────────────────────────
    epsilon:           float = 5.0
    delta:             float = 1e-5
    max_grad_norm:     float = 5.0
    dp_physical_batch_size: int = 4

    # ── Pilot mode ─────────────────────────────────────────────
    pilot_mode:        bool  = False
    pilot_train_size:  int   = 2500
    pilot_val_size:    int   = 250
    pilot_epochs:      int   = 1

    # ── Paths ──────────────────────────────────────────────────
    # STALE DEFAULTS: data_prep/Dataset/final/... does not exist in this
    # checkout (only data_prep_invalid/ does) -- these are leftover
    # defaults from an older dataset pipeline. Every MODELS entry except
    # llama3_2_1b_it inherits them unchanged, so any run of that model
    # config MUST override these four paths (via --train-path/--val-path/
    # --canary-path/--reference-path, or from canarybench.inject_canaries()'s
    # returned artifacts) or it will fail via require_dataset_paths() below
    # rather than silently training on the wrong/missing data.
    train_path:          str = repo_path("data_prep", "Dataset", "final", "combined_train.jsonl")
    val_path:            str = repo_path("data_prep", "Dataset", "final", "combined_val.jsonl")
    canary_path:         str = repo_path("data_prep", "Dataset", "final", "member_canaries.jsonl")
    reference_path:      str = repo_path("data_prep", "Dataset", "final", "attack_eval.jsonl")
    canary_registry_path:str = repo_path("data_prep", "data", "audit", "canary_registry.json")
    output_base:         str = repo_path("checkpoints")
    results_dir:         str = repo_path("results")

    # ── Canary eval settings ────────────────────────────────────
    canary_eval_samples: int = 500   # bumped from 200 — you have 7,250 unique members
    eval_canaries:       bool = True


def require_dataset_paths(cfg: "TrainingConfig") -> None:
    """Fail loudly, before any model/data loading, if this config's core
    dataset paths don't exist on disk.

    Does NOT guess or silently redirect to any particular dataset (e.g. it
    will not fall back to instruction_recall_dataset/dataset/processed/ or
    any other path) -- it only checks that whatever IS configured actually
    exists, and raises a clear, actionable error naming exactly which
    path(s) are missing if not.
    """
    required = {
        "train_path": cfg.train_path,
        "val_path": cfg.val_path,
        "canary_path": cfg.canary_path,
        "reference_path": cfg.reference_path,
    }
    missing = {name: path for name, path in required.items() if not os.path.exists(path)}
    if missing:
        lines = "\n".join(f"  --{name.replace('_', '-')}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            "TrainingConfig has dataset path(s) that do not exist on disk:\n"
            f"{lines}\n"
            "finetune/config.py's built-in defaults point at data_prep/Dataset/final/..., "
            "an older dataset pipeline not present in this checkout. Pass real paths "
            "explicitly via the CLI flags above, e.g. the InjectionArtifacts "
            "returned by canarybench.inject_canaries()."
        )


# ── Model registry — updated for 2x L40S (46GB each) ──────────────────────────
#
# Key principle: keep EFFECTIVE batch size = 64 across ALL runs
# This isolates DP noise effect from batch size differences
#
# Baseline (no DP):  batch_size × grad_accum × num_gpus = 64
# DP training:       batch_size × grad_accum = 64 (Opacus handles single-GPU)
#                    (Opacus doesn't support native multi-GPU, use FSDP or single)

MODELS = {
    "gemma_2b_it": TrainingConfig(
        model_name     = "google/gemma-2-2b-it",
        model_key      = "gemma_2b_it",
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        # Baseline: 8 × 4 × 2 GPUs = 64 effective
        # DP:       4 × 16 = 64 effective (single GPU, Opacus requirement)
        batch_size     = 4,
        grad_accum     = 16,
        dp_physical_batch_size = 4,
        learning_rate  = 1e-5,
        max_grad_norm  = 1.0,
        num_epochs     = 3,
    ),

    "vault_gemma": TrainingConfig(
        model_name     = "google/vaultgemma-1b",
        model_key      = "vault_gemma",
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        # 1B model — fits easily on L40S
        # Baseline: 16 × 2 × 2 GPUs = 64 effective
        # DP:       2 × 32 = 64 logical, split into physical batches of 2
        # Reduced batch/sequence for lower GPU memory usage
        batch_size     = 2,
        grad_accum     = 32,
        dp_physical_batch_size = 2,
        max_seq_length = 128,
        learning_rate  = 1e-5,
        max_grad_norm  = 1.0,
        num_epochs     = 3,
    ),

    "gemma3_1b": TrainingConfig(
        model_name     = "google/gemma-3-1b-it",
        model_key      = "gemma3_1b",
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        # Same size as VaultGemma — same batch sizes
        batch_size     = 8,
        grad_accum     = 8,
        dp_physical_batch_size = 2,
        learning_rate  = 1e-5,
        max_grad_norm  = 1.0,
        num_epochs     = 3,
    ),

    "llama3_2_1b_it": TrainingConfig(
        model_name     = "meta-llama/Llama-3.2-1B-Instruct",
        model_key      = "llama3_2_1b_it",
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        # Same scale as the other 1B instruction-tuned models.
        batch_size     = 8,
        grad_accum     = 8,
        dp_physical_batch_size = 8,
        learning_rate  = 1e-5,
        max_grad_norm  = 1.0,
        num_epochs     = 3,
        train_path     = repo_path("instruction_recall_dataset", "dataset", "processed", "combined_train.jsonl"),
        val_path       = repo_path("instruction_recall_dataset", "dataset", "processed", "combined_val.jsonl"),
        canary_path    = repo_path("instruction_recall_dataset", "dataset", "processed", "member_canaries.jsonl"),
        reference_path = repo_path("instruction_recall_dataset", "dataset", "processed", "attack_eval.jsonl"),
        canary_registry_path = repo_path("instruction_recall_dataset", "dataset", "processed", "canary_registry.json"),
        output_base    = repo_path("checkpoints", "instruction_recall"),
        results_dir    = repo_path("results", "instruction_recall"),
    ),
}

# ── DP epsilon values to sweep ─────────────────────────────────────────────────
EPSILON_VALUES = [3.0, 5.0, 8.0]

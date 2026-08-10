"""
Crash-safe checkpoint callback for DP training with canary evaluation.

This module provides the CrashSafeCallback class that handles:
- Recovery checkpoints every N steps
- Canary loss evaluation during training
- Epsilon target checkpointing
- Privacy audit logging
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

import torch
from opacus import PrivacyEngine


logger = logging.getLogger("dp_lora")


class CrashSafeCallback:
    """Saves recovery checkpoint every N steps and evaluates canaries during training.
    
    If run crashes, you restart from the last recovery checkpoint, not from zero.
    Also tracks canary losses and saves checkpoints at epsilon targets.
    """
    
    def __init__(self, privacy_engine: PrivacyEngine, delta: float = 1e-5,
                 member_canaries: Optional[List[Dict]] = None, 
                 reference_canaries: Optional[List[Dict]] = None,
                 save_every_steps: int = 500):
        """
        Args:
            privacy_engine: Opacus PrivacyEngine for epsilon tracking
            delta: Privacy delta for epsilon calculation
            member_canaries: List of member canary records with token_ids
            reference_canaries: List of reference canary records with token_ids
            save_every_steps: Save recovery checkpoint every N steps
        """
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.member_canaries = member_canaries or []
        self.reference_canaries = reference_canaries or []
        self.save_every = save_every_steps
        self.log_file = "privacy_audit_log.jsonl"
        self.prev_eps = 0.0
        
        # Ensure directories exist
        os.makedirs("checkpoints/recovery", exist_ok=True)
        os.makedirs("checkpoints/epsilon_targets", exist_ok=True)

    def on_step_end(self, state: Any, model: Optional[torch.nn.Module] = None):
        """Save recovery checkpoint every N steps."""
        if state.global_step % self.save_every == 0 and state.global_step > 0:
            path = f"checkpoints/recovery/step_{state.global_step}"
            model._module.save_pretrained(path)  # Save the underlying PEFT model
            
            # Save epsilon alongside so you know where you were
            try:
                eps = self.privacy_engine.get_epsilon(self.delta)
            except Exception as e:
                logger.warning(f"Failed to get epsilon: {e}")
                eps = -1
            
            with open(f"{path}/epsilon.json", "w") as f:
                json.dump({
                    "step": state.global_step, 
                    "epsilon": eps,
                    "epoch": state.epoch
                }, f)
            
            logger.info(f"[Recovery checkpoint] step={state.global_step} ε={eps:.4f}")

    def on_evaluate(self, state: Any, model: Optional[torch.nn.Module] = None):
        """Evaluate canaries and save epsilon target checkpoints."""
        try:
            eps = self.privacy_engine.get_epsilon(self.delta)
        except Exception as e:
            logger.warning(f"Failed to get epsilon: {e}")
            eps = -1

        # Compute canary losses
        member_losses, reference_losses = [], []
        if model is not None:
            model.eval()
            with torch.no_grad():
                for c in self.member_canaries:
                    ids = torch.tensor(c["token_ids"]).unsqueeze(0).to(model.device)
                    loss = model(ids, labels=ids).loss.item()
                    member_losses.append({
                        "id": c["id"], 
                        "loss": round(loss, 5),
                        "category": c.get("category", "unk"),
                        "repetitions": c.get("repetitions", 1)
                    })
                
                for c in self.reference_canaries:
                    ids = torch.tensor(c["token_ids"]).unsqueeze(0).to(model.device)
                    loss = model(ids, labels=ids).loss.item()
                    reference_losses.append({
                        "id": c["id"], 
                        "loss": round(loss, 5)
                    })

        # Save at epsilon targets
        for target in [3.0, 5.0, 8.0]:
            if self.prev_eps < target <= eps:
                path = f"checkpoints/epsilon_targets/eps_{target}"
                os.makedirs(path, exist_ok=True)
                model._module.save_pretrained(path)
                
                meta = {
                    "epsilon": eps, 
                    "target": target,
                    "epoch": state.epoch, 
                    "step": state.global_step,
                    "member_loss_mean": (
                        sum(x["loss"] for x in member_losses) 
                        / len(member_losses)
                    ) if member_losses else None,
                    "reference_loss_mean": (
                        sum(x["loss"] for x in reference_losses)
                        / len(reference_losses)
                    ) if reference_losses else None,
                }
                with open(f"{path}/meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                
                logger.info(f"[EPS TARGET] Saved ε={target} checkpoint → {path}")
        
        self.prev_eps = eps

        # Write audit log
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4),
            "epsilon_spent": round(eps, 5) if eps != -1 else None,
            "delta": self.delta,
            "member_loss_mean": (
                round(sum(x["loss"] for x in member_losses) 
                      / len(member_losses), 5)
            ) if member_losses else None,
            "reference_loss_mean": (
                round(sum(x["loss"] for x in reference_losses)
                      / len(reference_losses), 5)
            ) if reference_losses else None,
            "member_canary_losses": member_losses,
            "reference_canary_losses": reference_losses,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"[Privacy audit] ε={eps:.4f} | "
                   f"member={record['member_loss_mean']} | "
                   f"ref={record['reference_loss_mean']}")


def load_canaries_for_checkpointing(canary_path: str, reference_path: str, 
                                   tokenizer: Any, max_samples: int = 200) -> tuple:
    """Load canaries for crash-safe checkpointing.
    
    Args:
        canary_path: Path to member canaries file
        reference_path: Path to reference canaries file  
        tokenizer: Tokenizer for encoding canaries
        max_samples: Maximum number of canaries to load
        
    Returns:
        Tuple of (member_canaries, reference_canaries)
    """
    member_canaries = []
    reference_canaries = []
    
    # Load member canaries
    if os.path.exists(canary_path):
        with open(canary_path) as f:
            member_records = [json.loads(l) for l in f if l.strip()]
        for rec in member_records[:max_samples]:
            if rec.get("task") == "lm" or not rec.get("input", "").strip():
                full = rec["target"]
            else:
                prompt = rec["input"] + "\n\n### Response:\n"
                full = prompt + rec["target"]
            enc = tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            member_canaries.append({
                "id": rec.get("id", ""),
                "token_ids": enc["input_ids"][0].tolist(),
                "category": rec.get("category", "unk"),
                "repetitions": rec.get("repetitions", 1)
            })
        logger.info(f"Loaded {len(member_canaries)} member canaries")
    
    # Load reference canaries  
    if os.path.exists(reference_path):
        with open(reference_path) as f:
            reference_records = [json.loads(l) for l in f if l.strip()]
        for rec in reference_records[:max_samples]:
            if rec.get("task") == "lm" or not rec.get("input", "").strip():
                full = rec["target"]
            else:
                prompt = rec["input"] + "\n\n### Response:\n"
                full = prompt + rec["target"]
            enc = tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            reference_canaries.append({
                "id": rec.get("id", ""),
                "token_ids": enc["input_ids"][0].tolist()
            })
        logger.info(f"Loaded {len(reference_canaries)} reference canaries")
    
    return member_canaries, reference_canaries

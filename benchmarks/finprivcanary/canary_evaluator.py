"""
Canary evaluation utilities for DP training.

This module provides functions to evaluate canary losses during training
and compute various metrics for privacy analysis.
"""

import os
import sys
import json
from typing import List, Dict, Optional, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from canarybench.prompt_format import build_supervised_text, legacy_hardcoded_supervised_text


def compute_loss_statistics(member_losses: List[Dict], reference_losses: List[Dict]) -> Dict[str, float]:
    """
    Compute comprehensive loss statistics for canary analysis.
    
    Args:
        member_losses: List of member canary losses
        reference_losses: List of reference canary losses
        
    Returns:
        Dictionary of loss statistics
    """
    member_vals = [x["loss"] for x in member_losses]
    reference_vals = [x["loss"] for x in reference_losses]
    
    stats = {}
    
    if member_vals:
        stats["member_mean"] = sum(member_vals) / len(member_vals)
        stats["member_median"] = sorted(member_vals)[len(member_vals) // 2]
        stats["member_min"] = min(member_vals)
        stats["member_max"] = max(member_vals)
    
    if reference_vals:
        stats["reference_mean"] = sum(reference_vals) / len(reference_vals)
        stats["reference_median"] = sorted(reference_vals)[len(reference_vals) // 2]
        stats["reference_min"] = min(reference_vals)
        stats["reference_max"] = max(reference_vals)
    
    if member_vals and reference_vals:
        stats["gap_mean"] = stats["reference_mean"] - stats["member_mean"]
        stats["gap_median"] = stats["reference_median"] - stats["member_median"]
    
    return stats


def save_canary_evaluation_results(output_path: str, results: Dict[str, Any]) -> str:
    """
    Save canary evaluation results to JSON file.
    
    Args:
        output_path: Path to save results
        results: Dictionary of evaluation results
        
    Returns:
        Path where results were saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_path


def load_canary_records(canary_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load canary records from file.
    
    Args:
        canary_path: Path to canary file
        max_samples: Maximum number of records to load
        
    Returns:
        List of canary records
    """
    if not os.path.exists(canary_path):
        return []
    
    with open(canary_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    if max_samples:
        records = records[:max_samples]
    
    return records


def encode_canary_for_evaluation(canary_record: Dict, tokenizer: Any, prompt_mode: str = "canonical") -> Dict:
    """
    Encode a single canary record for evaluation.

    Args:
        canary_record: Canary record with input and target
        tokenizer: The tokenizer
        prompt_mode: "canonical" (default, chat-template-aware, matches training)
            or "legacy_hardcoded" (old unconditional "### Response:\n" template,
            for reproducing historical results exactly). Previously this
            function always used the legacy template regardless of tokenizer;
            this function has no callers anywhere in the current repo (dead
            code), so the default change here has no observed effect on any
            existing pipeline.

    Returns:
        Encoded canary with token_ids
    """
    if prompt_mode == "legacy_hardcoded":
        _, full = legacy_hardcoded_supervised_text(canary_record)
    else:
        _, full = build_supervised_text(tokenizer, canary_record)
    enc = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    return {
        "id": canary_record.get("id", ""),
        "token_ids": enc["input_ids"][0].tolist(),
        "field_type": canary_record.get("field_type", "unk"),
        "exposure": canary_record.get("exposure", 1),
    }

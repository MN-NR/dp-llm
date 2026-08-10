from typing import Any, Dict, List, Optional

from metrics.base import load_jsonl, safe_mean


def summarize_privacy_audit_log(log_path: str) -> Dict[str, Any]:
    records = load_jsonl(log_path)
    if not records:
        return {
            "num_records": 0,
            "epsilon_final": None,
            "delta": None,
            "member_loss_mean": None,
            "reference_loss_mean": None,
        }

    last = records[-1]
    member_means = [r["member_loss_mean"] for r in records if r.get("member_loss_mean") is not None]
    reference_means = [r["reference_loss_mean"] for r in records if r.get("reference_loss_mean") is not None]

    return {
        "num_records": len(records),
        "epsilon_final": last.get("epsilon_spent"),
        "delta": last.get("delta"),
        "member_loss_mean": safe_mean(member_means),
        "reference_loss_mean": safe_mean(reference_means),
        "steps": [r.get("step") for r in records],
        "epsilons": [r.get("epsilon_spent") for r in records],
    }


def extract_epsilon_trace(log_path: str) -> List[Dict[str, Optional[float]]]:
    return [
        {
            "step": record.get("step"),
            "epoch": record.get("epoch"),
            "epsilon": record.get("epsilon_spent"),
            "delta": record.get("delta"),
        }
        for record in load_jsonl(log_path)
    ]


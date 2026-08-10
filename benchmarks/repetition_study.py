from typing import Any, Dict, List, Optional, Sequence

from metrics.base import load_jsonl, save_json
from metrics.privacy.memorization import (
    compute_canary_losses,
    exposure_by_category,
    exposure_by_repetition,
)


def run_repetition_study(
    model: Any,
    tokenizer: Any,
    member_path: str,
    reference_path: str,
    device: str = "cuda",
    max_member_samples: Optional[int] = 500,
    max_reference_samples: Optional[int] = 1000,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    member_canaries = load_jsonl(member_path, max_samples=max_member_samples)
    reference_canaries = load_jsonl(reference_path, max_samples=max_reference_samples)

    member_losses = compute_canary_losses(model, tokenizer, member_canaries, device=device)
    reference_losses = compute_canary_losses(model, tokenizer, reference_canaries, device=device)

    result = {
        "metadata_validation": {
            "has_category_metadata": any(item.get("category") is not None for item in member_canaries),
            "has_repetition_metadata": any(item.get("repetitions") is not None for item in member_canaries),
        },
        "exposure_by_repetition": exposure_by_repetition(member_losses, reference_losses),
        "exposure_by_category": exposure_by_category(member_losses, reference_losses),
        "member_losses": member_losses,
        "reference_losses": reference_losses,
    }
    if output_path:
        save_json(output_path, result)
    return result

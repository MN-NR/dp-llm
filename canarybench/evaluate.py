"""evaluate(): thin wrapper around the canonical metrics/privacy/* and
metrics/utility/* implementations. Reimplements no metric formulas.

  gap / exposure_mean / exposure_p50 / exposure_p95  -> metrics.privacy.memorization
  WBC AUC / bootstrap CI                             -> metrics.privacy.inference
  extraction (one greedy completion per canary)       -> metrics.privacy.extraction (UNCHANGED)

The one thing this module adds on top of the canonical metrics is a canary
*selection* layer for extraction, matching what the paper called "targeted
top-20": that name referred to *which 20 canaries* get one greedy
extraction attempt each (the 20 most-memorized, i.e. lowest-loss, canaries
within a frequency tier) -- not a top-k decoding strategy. The decoding
itself is still exactly metrics.privacy.extraction.pii_extraction_rate's
existing single greedy completion (do_sample=False), unchanged.
"""

import random as _random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from metrics.base import load_causal_lm, load_jsonl
from metrics.privacy.extraction import pii_extraction_rate
from metrics.privacy.inference import run_wbc_attack
from metrics.privacy.memorization import compute_canary_exposure, memorization_by_frequency, memorization_by_pii_type

DEFAULT_FREQUENCY_TIERS = (1, 10, 50)


@dataclass
class EvaluationResult:
    checkpoint: str
    base_model: str
    exposure: Dict[str, Any]
    wbc: Optional[Dict[str, Any]]
    targeted_extraction: Dict[str, Any]
    frequency_breakdown: Dict[str, Any]
    pii_type_breakdown: Dict[str, Any]
    mask_prompt: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "base_model": self.base_model,
            "exposure": self.exposure,
            "wbc": self.wbc,
            "targeted_extraction": self.targeted_extraction,
            "frequency_breakdown": self.frequency_breakdown,
            "pii_type_breakdown": self.pii_type_breakdown,
            "mask_prompt": self.mask_prompt,
        }


def select_targets(
    member_losses: Sequence[Dict[str, Any]],
    n_targets: int = 20,
    target_selection: str = "most_memorized",
    frequency_tier: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Selects which canaries get an extraction attempt. This is the
    "targeted top-20" selection step -- ranking/filtering, not decoding.

    target_selection:
      "most_memorized" (default) -- lowest loss first, matches the paper.
      "random" -- for later most-memorized-N vs random-N comparisons.
    """
    pool = list(member_losses)
    if frequency_tier is not None and pool:
        freq_key = "frequency" if "frequency" in pool[0] else "exposure"
        pool = [r for r in pool if r.get(freq_key) == frequency_tier]

    if target_selection == "most_memorized":
        ranked = sorted(pool, key=lambda r: r["loss"])
    elif target_selection == "random":
        ranked = list(pool)
        _random.Random(seed).shuffle(ranked)
    else:
        raise ValueError(f"Unknown target_selection: {target_selection!r}")

    return ranked[:n_targets]


def run_targeted_extraction(
    model: Any,
    tokenizer: Any,
    member_losses: Sequence[Dict[str, Any]],
    device: str,
    n_targets: int = 20,
    target_selection: str = "most_memorized",
    frequency_tiers: Sequence[int] = DEFAULT_FREQUENCY_TIERS,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for tier in frequency_tiers:
        targets = select_targets(
            member_losses, n_targets=n_targets, target_selection=target_selection, frequency_tier=tier
        )
        if not targets:
            continue
        # Unchanged: one greedy completion per canary (do_sample=False).
        extraction = pii_extraction_rate(model, tokenizer, targets, device)
        results[f"{tier}x"] = {
            "target_selection": target_selection,
            "n_targets": len(targets),
            "decoding": "greedy",
            "num_reproduced_exactly": round(extraction["pii_extraction_rate"] * extraction["total_attempts"]),
            **extraction,
        }
    return results


def evaluate(
    base_model_path: str,
    adapter_path: Optional[str],
    artifacts: Any,
    device: str = "cuda",
    max_member_samples: int = 770,
    max_reference_samples: int = 1000,
    reference_model_path: Optional[str] = None,
    run_wbc: bool = True,
    run_extraction: bool = True,
    n_extraction_targets: int = 20,
    mask_prompt: bool = False,
) -> EvaluationResult:
    """artifacts: an object with .member_path / .reference_path (e.g.
    canarybench.inject.InjectionArtifacts, or any object with those two
    attributes pointing at member_canaries.jsonl / attack_eval.jsonl-shaped
    files).

    mask_prompt default is False (full-sequence loss, prompt included --
    only padding tokens excluded), NOT metrics.privacy.memorization's own
    module-level default of True (target-only). This is a deliberate,
    empirically-resolved choice, not an oversight:

    Reproducing the paper's reported VaultGemma baseline numbers
    (memorization gap ~=0.6256, WBC AUC ~=0.7455) against
    checkpoints/vault_gemma/baseline/final only worked with mask_prompt=False:
      mask_prompt=True  (target-only): gap=1.2403  (does not match)
      mask_prompt=False (full-sequence): gap=0.6233  (matches within 0.4%)
    This also explains why WBC AUC (0.7433 vs 0.7455, matches within 0.3%)
    reproduced correctly from the start: metrics.privacy.inference.run_wbc_attack
    hardcodes mask_prompt=False for its token losses already. Corroborated by
    rebuttal_experiments/eval_missing_canaries.py -- the actual paper-table
    generator for gap_by_frequency -- which computes loss with only padding
    masked (no prompt masking) at all. Pass mask_prompt=True explicitly if you
    want the target-only definition instead; it will not match the paper.
    """
    loaded = load_causal_lm(base_model_path=base_model_path, adapter_path=adapter_path, device=device)
    member_canaries = load_jsonl(artifacts.member_path, max_samples=max_member_samples)
    reference_canaries = load_jsonl(artifacts.reference_path, max_samples=max_reference_samples)

    exposure = compute_canary_exposure(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        member_canaries=member_canaries,
        reference_canaries=reference_canaries,
        device=device,
        mask_prompt=mask_prompt,
    )
    frequency_breakdown = memorization_by_frequency(exposure["member_losses"], exposure["reference_loss_mean"])
    pii_type_breakdown = memorization_by_pii_type(exposure["member_losses"], exposure["reference_loss_mean"])

    wbc = None
    if run_wbc:
        reference_loaded = None
        if reference_model_path:
            reference_loaded = load_causal_lm(base_model_path=reference_model_path, adapter_path=None, device=device)
        wbc = run_wbc_attack(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            member_canaries=member_canaries,
            reference_canaries=reference_canaries,
            device=device,
            reference_model=reference_loaded.model if reference_loaded else None,
            reference_tokenizer=reference_loaded.tokenizer if reference_loaded else None,
        )

    targeted_extraction: Dict[str, Any] = {}
    if run_extraction:
        targeted_extraction = run_targeted_extraction(
            loaded.model,
            loaded.tokenizer,
            exposure["member_losses"],
            device,
            n_targets=n_extraction_targets,
        )

    return EvaluationResult(
        checkpoint=adapter_path or base_model_path,
        base_model=base_model_path,
        exposure=exposure,
        wbc=wbc,
        targeted_extraction=targeted_extraction,
        frequency_breakdown=frequency_breakdown,
        pii_type_breakdown=pii_type_breakdown,
        mask_prompt=mask_prompt,
    )

from typing import Any, Dict, Mapping, Optional

from metrics.base import load_causal_lm, load_jsonl, save_json
from metrics.composite.privacy_composite import summarize_privacy_profile
from metrics.privacy.extraction import pii_extraction_rate
from metrics.privacy.inference import run_wbc_attack
from metrics.privacy.memorization import compute_canary_exposure
from metrics.utility.lm import compute_perplexity
from metrics.utility.task import evaluate_group_a

try:  # Optional metric (not required for paper)
    from metrics.utility.lm import group_c_reconstruction  # type: ignore
except Exception:  # pragma: no cover
    group_c_reconstruction = None


def evaluate_checkpoint(
    base_model_path: str,
    adapter_path: Optional[str],
    member_path: str,
    reference_path: str,
    val_path: str,
    device: str = "cuda",
    baseline_gap: Optional[float] = None,
    max_member_samples: int = 770,
    max_reference_samples: int = 1000,
    reference_model_path: Optional[str] = None,
    wbc_window_lengths: Optional[list] = None,
    include_group_a: bool = False,
    include_group_c: bool = False,
    debug_model_loading: bool = False,
) -> Dict[str, Any]:
    loaded = load_causal_lm(base_model_path=base_model_path, adapter_path=adapter_path, device=device)
    if debug_model_loading:
        print("EVAL MODEL:", base_model_path)
        print("ADAPTER:", adapter_path)
    reference_loaded = load_causal_lm(
        base_model_path=reference_model_path or base_model_path,
        adapter_path=None,
        device=device,
    )
    if debug_model_loading:
        print("REFERENCE MODEL:", reference_model_path or base_model_path)
    member_canaries = load_jsonl(member_path, max_samples=max_member_samples)
    reference_canaries = load_jsonl(reference_path, max_samples=max_reference_samples)

    exposure = compute_canary_exposure(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        member_canaries=member_canaries,
        reference_canaries=reference_canaries,
        device=device,
    )
    wbc = run_wbc_attack(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        member_canaries=member_canaries,
        reference_canaries=reference_canaries,
        device=device,
        reference_model=reference_loaded.model,
        reference_tokenizer=reference_loaded.tokenizer,
        require_reference_model=True,
        window_lengths=wbc_window_lengths,
    )
    if debug_model_loading:
        print("\n=== DEBUG ===")
        print("ADAPTER:", adapter_path)
        print("first member id:", (member_canaries[0].get("id") if member_canaries else None))
        print("first reference id:", (reference_canaries[0].get("id") if reference_canaries else None))
        print("member mean:", exposure.get("member_loss_mean"))
        print("reference mean:", exposure.get("reference_loss_mean"))
        print("gap:", exposure.get("gap"))
        print("exp mean:", exposure.get("exposure_mean"))
        print("exp p95:", exposure.get("exposure_p95"))
        print("wbc:", wbc)
        print("==============\n")
    utility = evaluate_group_a(loaded.model, loaded.tokenizer, val_path, device=device) if include_group_a else None
    perplexity = compute_perplexity(loaded.model, loaded.tokenizer, val_path, device=device)
    if include_group_c:
        if group_c_reconstruction is None:
            raise ImportError(
                "Requested Group C metric (`include_group_c=True`), but `group_c_reconstruction` "
                "is not available in `metrics.utility.lm`."
            )
        group_c = group_c_reconstruction(loaded.model, loaded.tokenizer, val_path, device=device)
    else:
        group_c = None
    extraction = pii_extraction_rate(loaded.model, loaded.tokenizer, member_canaries, device=device)

    return {
        "canary_exposure": exposure,
        "wbc": wbc,
        "utility": utility,
        "perplexity": perplexity,
        "group_c": group_c,
        "pii_extraction": extraction,
        "privacy_composite": summarize_privacy_profile(
            exposure_result=exposure,
            wbc_result=wbc,
            extraction_result=extraction,
            baseline_gap=baseline_gap,
        ),
    }


def evaluate_checkpoints(
    checkpoints: Mapping[str, Dict[str, Any]],
    member_path: str,
    reference_path: str,
    val_path: str,
    device: str = "cuda",
    output_path: Optional[str] = None,
    debug_model_loading: bool = False,
) -> Dict[str, Any]:
    all_results: Dict[str, Any] = {}
    for name, config in checkpoints.items():
        all_results[name] = evaluate_checkpoint(
            base_model_path=config["base"],
            adapter_path=config.get("adapter"),
            member_path=member_path,
            reference_path=reference_path,
            val_path=val_path,
            device=device,
            baseline_gap=config.get("baseline_gap"),
            max_member_samples=config.get("max_member_samples", 770),
            max_reference_samples=config.get("max_reference_samples", 1000),
            reference_model_path=config.get("reference_model_path"),
            wbc_window_lengths=config.get("wbc_window_lengths"),
            include_group_a=bool(config.get("include_group_a", False)),
            include_group_c=bool(config.get("include_group_c", False)),
            debug_model_loading=debug_model_loading,
        )
        all_results[name]["epsilon"] = config.get("epsilon")
    if output_path:
        save_json(output_path, all_results)
    return all_results

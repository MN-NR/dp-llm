"""
Run the full dp-llm privacy/utility profile over a standard checkpoint matrix.

This script evaluates:
  - baselines (adapter under `baseline/final`)
  - DP runs for eps in {3,5,8} (adapter under `dp_eps{eps}.0/{best,final}`)

It writes one JSON per run under `results_testing/profile_matrix/`.
"""

from __future__ import annotations

import sys
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_PY_ROOT))

from benchmarks.cross_model_exposure import evaluate_checkpoint  # noqa: E402
from metrics.base import save_json  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_base: str
    ckpt_dir: Path


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(key="gemma3_1b", hf_base="google/gemma-3-1b-it", ckpt_dir=Path("checkpoints/gemma3_1b")),
    ModelSpec(key="vault_gemma", hf_base="google/vaultgemma-1b", ckpt_dir=Path("checkpoints/vault_gemma")),
    ModelSpec(key="gemma_2b_it", hf_base="google/gemma-2-2b-it", ckpt_dir=Path("checkpoints/gemma_2b_it")),
    ModelSpec(
        key="llama3_2_1b_it",
        hf_base="meta-llama/Llama-3.2-1B-Instruct",
        ckpt_dir=Path("checkpoints/instruction_recall/llama3_2_1b_it"),
    ),
]


def _pick_adapter_dir(root: Path) -> Optional[Path]:
    """
    Prefer `best/`, then `final/`, else None.
    """
    for name in ("best", "final"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ckpt_dir(spec: ModelSpec) -> Path:
    """
    Resolve checkpoint dir relative to repo root so the script works
    regardless of the current working directory.
    """
    if spec.ckpt_dir.is_absolute():
        return spec.ckpt_dir
    return (_REPO_PY_ROOT / spec.ckpt_dir).resolve()


def _resolve_data_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((_REPO_PY_ROOT / path).resolve())


def _iter_runs(spec: ModelSpec, eps_values: Iterable[float]) -> Iterable[tuple[str, Optional[float], Path]]:
    ckpt_dir = _resolve_ckpt_dir(spec)
    baseline_adapter = _pick_adapter_dir(ckpt_dir / "baseline")
    if baseline_adapter is not None:
        yield ("baseline", None, baseline_adapter)

    for eps in eps_values:
        eps_dir = ckpt_dir / f"dp_eps{eps:.1f}"
        adapter = _pick_adapter_dir(eps_dir)
        if adapter is not None:
            yield (f"dp_eps{eps:.1f}", eps, adapter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results_testing/profile_matrix")
    parser.add_argument("--models", default="gemma3_1b,vault_gemma,gemma_2b_it,llama3_2_1b_it")
    parser.add_argument("--eps", default="3,5,8", help="Comma-separated eps values (e.g. 3,5,8).")
    parser.add_argument("--include-group-a", action="store_true", help="Also run Group A downstream utility.")
    parser.add_argument("--include-group-c", action="store_true", help="Also run Group C reconstruction metric.")
    parser.add_argument("--member-path", default="instruction_recall_dataset/dataset/processed/member_canaries.jsonl")
    parser.add_argument("--reference-path", default="instruction_recall_dataset/dataset/processed/attack_eval.jsonl")
    parser.add_argument("--val-path", default="instruction_recall_dataset/dataset/processed/combined_val.jsonl")
    parser.add_argument("--max-member-samples", type=int, default=770)
    parser.add_argument("--max-reference-samples", type=int, default=1000)
    args = parser.parse_args()

    requested = {m.strip() for m in str(args.models).split(",") if m.strip()}
    eps_values = [float(x.strip()) for x in str(args.eps).split(",") if x.strip()]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Any] = {"out_dir": str(out_root), "runs": []}

    for spec in DEFAULT_MODELS:
        if spec.key not in requested:
            continue

        member_path = _resolve_data_path(args.member_path)
        reference_path = _resolve_data_path(args.reference_path)
        val_path = _resolve_data_path(args.val_path)

        runs = list(_iter_runs(spec, eps_values))
        if not runs:
            ckpt_dir = _resolve_ckpt_dir(spec)
            print(f"[skip] {spec.key}: no runs found under {ckpt_dir} (eps={eps_values})")
            continue

        for run_name, eps, adapter_dir in runs:
            out_path = out_root / f"{spec.key}__{run_name}.json"
            print(f"[eval] {spec.key} / {run_name} -> {out_path}")

            results = evaluate_checkpoint(
                base_model_path=spec.hf_base,
                adapter_path=str(adapter_dir),
                member_path=member_path,
                reference_path=reference_path,
                val_path=val_path,
                device=args.device,
                max_member_samples=args.max_member_samples,
                max_reference_samples=args.max_reference_samples,
                baseline_gap=None,
                reference_model_path=spec.hf_base,
                include_group_c=bool(args.include_group_c),
                include_group_a=bool(args.include_group_a),
            )
            results["model_key"] = spec.key
            results["run"] = run_name
            results["epsilon"] = eps
            results["base_model_path"] = spec.hf_base
            results["adapter_path"] = str(adapter_dir)

            save_json(str(out_path), results)
            index["runs"].append({"model_key": spec.key, "run": run_name, "epsilon": eps, "path": str(out_path)})

    save_json(str(out_root / "index.json"), index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

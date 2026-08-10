import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any
from pathlib import Path

SCRIPT_DIR = os.path.dirname(__file__)
ALT_PROFILE_DIR = str((Path(SCRIPT_DIR) / ".." / ".." / ".." / "results_testing" / "profile_matrix").resolve())
REPO_PY_ROOT = (Path(SCRIPT_DIR) / ".." / "..").resolve()


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _iter_result_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []

    index_candidates = [args.index] if args.index else [os.path.join(SCRIPT_DIR, "index.json"), os.path.join(ALT_PROFILE_DIR, "index.json")]
    for index_path in index_candidates:
        if not index_path or not os.path.exists(index_path):
            continue
        index = _load_json(index_path)
        runs = index.get("runs", [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            if isinstance(run, dict) and isinstance(run.get("path"), str):
                p = run["path"]
                if os.path.isabs(p):
                    paths.append(p)
                    continue

                index_dir = Path(index_path).resolve().parent
                candidate = (index_dir / p).resolve()
                if candidate.exists():
                    paths.append(str(candidate))
                    continue

                candidate = (REPO_PY_ROOT / p).resolve()
                if candidate.exists():
                    paths.append(str(candidate))
                    continue

                paths.append(str((index_dir / p).resolve()))

    patterns = [args.glob] if args.glob else [os.path.join(SCRIPT_DIR, "*.json"), os.path.join(ALT_PROFILE_DIR, "*.json")]
    for pattern in patterns:
        if not pattern:
            continue
        paths.extend(glob.glob(pattern))

    # Deduplicate, keep stable order
    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(paths):
        if os.path.basename(p) == "index.json":
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-frequency and field-type breakdown of member losses.")
    parser.add_argument("--glob", default=None, help="Glob for result JSON files (fallback if no index.json).")
    parser.add_argument("--index", default=None, help="Path to index.json (defaults to ./index.json).")
    args = parser.parse_args()

    paths = _iter_result_paths(args)
    printed = 0
    for path in paths:
        r = _load_json(path)

        model = r.get("model_key") or os.path.basename(path).split("__", 1)[0]
        eps = r.get("epsilon", None)
        if eps is None:
            eps = "∞"

        ce = r.get("canary_exposure", {}) if isinstance(r.get("canary_exposure"), dict) else {}
        ref_mean = float(ce.get("reference_loss_mean", 0) or 0)

        member_losses = ce.get("member_losses", [])
        if not isinstance(member_losses, list) or not member_losses:
            continue

        printed += 1
        print(f"\n{'='*50}")
        print(f"{model} ε={eps}")
        print(f"{'='*50}")

        # Per-frequency breakdown
        freq_groups: dict[int, list[float]] = defaultdict(list)
        for m in member_losses:
            if not isinstance(m, dict):
                continue
            freq = int(m.get("exposure", 1) or 1)
            loss = m.get("loss", None)
            if loss is None:
                continue
            freq_groups[freq].append(float(loss))

        print("\n  By frequency:")
        for freq in sorted(freq_groups):
            losses = freq_groups[freq]
            mem_mean = sum(losses) / len(losses)
            gap = ref_mean - mem_mean
            print(f"    {freq}x: n={len(losses):>4}  mem_loss={mem_mean:.4f}  gap={gap:+.4f}")

        # Per-field-type breakdown
        type_groups: dict[str, list[float]] = defaultdict(list)
        for m in member_losses:
            if not isinstance(m, dict):
                continue
            ft = str(m.get("field_type", "unknown") or "unknown")
            loss = m.get("loss", None)
            if loss is None:
                continue
            type_groups[ft].append(float(loss))

        print("\n  By field type:")
        for ft in sorted(type_groups):
            losses = type_groups[ft]
            mem_mean = sum(losses) / len(losses)
            gap = ref_mean - mem_mean
            print(f"    {ft:<15}: n={len(losses):>4}  mem_loss={mem_mean:.4f}  gap={gap:+.4f}")

    if printed == 0:
        print("(no runs had canary_exposure.member_losses)")
        print(f"Tried index: {args.index or os.path.join(SCRIPT_DIR, 'index.json')}")
        print(f"Tried glob:  {args.glob or os.path.join(SCRIPT_DIR, '*.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

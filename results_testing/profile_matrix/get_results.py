# import argparse
# import glob
# import json
# import os
# from typing import Any

# SCRIPT_DIR = os.path.dirname(__file__)
# INNER_PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
# REPO_ROOT = os.path.abspath(os.path.join(INNER_PKG_ROOT, ".."))


# def _resolve_path(path: str) -> str:
#     """
#     Resolve paths from index.json robustly regardless of current working directory.

#     Some index entries are stored relative to the repo root (e.g. "dp-llm/...").
#     When running this script from within `profile_matrix/`, those relative paths
#     would otherwise be interpreted relative to the CWD and fail.
#     """
#     if os.path.isabs(path):
#         return path

#     candidates = (
#         path,
#         os.path.join(SCRIPT_DIR, path),
#         os.path.join(INNER_PKG_ROOT, path),
#         os.path.join(REPO_ROOT, path),
#     )
#     for candidate in candidates:
#         if os.path.exists(candidate):
#             return candidate
#     return path


# def _load_json(path: str) -> dict[str, Any]:
#     resolved = _resolve_path(path)
#     with open(resolved) as fh:
#         data = json.load(fh)
#     if not isinstance(data, dict):
#         raise ValueError(f"Expected JSON object in {resolved}, got {type(data).__name__}")
#     return data


# def _iter_result_paths(args: argparse.Namespace) -> list[str]:
#     # Prefer index.json if it exists because it’s explicit about which runs belong in the matrix.
#     index_path = args.index or os.path.join(SCRIPT_DIR, "index.json")
#     fallback_pattern = args.glob or os.path.join(SCRIPT_DIR, "*.json")
#     fallback_paths = [
#         p for p in sorted(glob.glob(fallback_pattern)) if os.path.basename(p) != "index.json"
#     ]
#     if os.path.exists(index_path):
#         index = _load_json(index_path)
#         runs = index.get("runs", [])
#         if isinstance(runs, list):
#             paths: list[str] = []
#             for run in runs:
#                 if isinstance(run, dict) and isinstance(run.get("path"), str):
#                     paths.append(run["path"])
#             if paths:
#                 # Union index + glob so partial index files don't hide new runs (e.g. eps3/5/8).
#                 seen: set[str] = set()
#                 merged: list[str] = []
#                 for p in paths + fallback_paths:
#                     rp = _resolve_path(p)
#                     if rp not in seen and os.path.exists(rp):
#                         seen.add(rp)
#                         merged.append(rp)
#                 if merged:
#                     return merged

#     return fallback_paths


# def main() -> int:
#     parser = argparse.ArgumentParser(description="Print a summary table for profile_matrix results.")
#     parser.add_argument("--glob", default=None, help="Glob for result JSON files (fallback if no index.json).")
#     parser.add_argument("--index", default=None, help="Path to index.json (defaults to ./index.json).")
#     args = parser.parse_args()

#     rows = []
#     paths = _iter_result_paths(args)
#     for path in paths:
#         r = _load_json(path)

#         ce = r.get("canary_exposure", {}) if isinstance(r.get("canary_exposure"), dict) else {}
#         wbc = r.get("wbc", {}) if isinstance(r.get("wbc"), dict) else {}

#         epsilon = r.get("epsilon", None)
#         if epsilon is None:
#             epsilon = "∞"

#         rows.append(
#             {
#                 "model": r.get("model_key") or os.path.basename(path).split("__", 1)[0],
#                 "epsilon": epsilon,
#                 "gap": round(float(ce.get("gap", 0) or 0), 4),
#                 "exposure_mean": round(float(ce.get("exposure_mean", 0) or 0), 2),
#                 "exposure_p95": round(float(ce.get("exposure_p95", 0) or 0), 2),
#                 "exposure_max": round(float(ce.get("exposure_max", 0) or 0), 2),
#                 "wbc_auc": float(wbc.get("wbc_auc", 0) or 0),
#             }
#         )

#     # Print table
#     print(f"{'Model':<15} {'ε':<6} {'Gap':>8} {'Exp Mean':>9} {'Exp P95':>8} {'WBC AUC':>8}")
#     print("-" * 60)
#     for r in sorted(rows, key=lambda x: (str(x["model"]), str(x["epsilon"]))):
#         print(
#             f"{r['model']:<15} {str(r['epsilon']):<6} {r['gap']:>+8.4f} "
#             f"{r['exposure_mean']:>9.2f} {r['exposure_p95']:>8.2f} {r['wbc_auc']:>8.4f}"
#         )
#     if not rows:
#         print("(no result JSONs found)")
#         print(f"Tried index: {args.index or os.path.join(SCRIPT_DIR, 'index.json')}")
#         print(f"Tried glob:  {args.glob or os.path.join(SCRIPT_DIR, '*.json')}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
import argparse
import glob
import json
import os
from typing import Any

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PARENT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))


def _resolve_path(path: str) -> str:
    """
    Resolve paths from index.json robustly regardless of current working directory.

    Some index entries are stored relative to the repo root. When running this
    script from within `profile_matrix/`, those relative paths would otherwise
    be interpreted relative to the CWD and fail.
    """
    if os.path.isabs(path):
        return path

    candidates = (
        path,
        os.path.join(SCRIPT_DIR, path),
        os.path.join(REPO_ROOT, path),
        os.path.join(PARENT_ROOT, path),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _load_json(path: str) -> dict[str, Any]:
    resolved = _resolve_path(path)
    with open(resolved) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {resolved}, got {type(data).__name__}")
    return data


def _iter_result_paths(args: argparse.Namespace) -> list[str]:
    # Prefer index.json if it exists because it’s explicit about which runs belong in the matrix.
    index_path = args.index or os.path.join(SCRIPT_DIR, "index.json")
    fallback_pattern = args.glob or os.path.join(SCRIPT_DIR, "*.json")
    fallback_paths = [
        p for p in sorted(glob.glob(fallback_pattern)) if os.path.basename(p) != "index.json"
    ]
    if os.path.exists(index_path):
        index = _load_json(index_path)
        runs = index.get("runs", [])
        if isinstance(runs, list):
            paths: list[str] = []
            for run in runs:
                if isinstance(run, dict) and isinstance(run.get("path"), str):
                    paths.append(run["path"])
            if paths:
                # Union index + glob so partial index files don't hide new runs (e.g. eps3/5/8).
                seen: set[str] = set()
                merged: list[str] = []
                for p in paths + fallback_paths:
                    rp = _resolve_path(p)
                    if rp not in seen and os.path.exists(rp):
                        seen.add(rp)
                        merged.append(rp)
                if merged:
                    return merged

    return fallback_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a summary table for profile_matrix results.")
    parser.add_argument("--glob", default=None, help="Glob for result JSON files (fallback if no index.json).")
    parser.add_argument("--index", default=None, help="Path to index.json (defaults to ./index.json).")
    args = parser.parse_args()

    rows = []
    paths = _iter_result_paths(args)
    for path in paths:
        r = _load_json(path)

        ce = r.get("canary_exposure", {}) if isinstance(r.get("canary_exposure"), dict) else {}
        wbc = r.get("wbc", {}) if isinstance(r.get("wbc"), dict) else {}
        recall = r.get("secret_recall", {}) if isinstance(r.get("secret_recall"), dict) else {}
        member_recall = recall.get("member", {}) if isinstance(recall.get("member"), dict) else {}
        reference_recall = recall.get("reference", {}) if isinstance(recall.get("reference"), dict) else {}
        perplexity = r.get("perplexity", {}) if isinstance(r.get("perplexity"), dict) else {}

        epsilon = r.get("epsilon", None)
        if epsilon is None:
            epsilon = "∞"

        rows.append(
            {
                "model": r.get("model_key") or os.path.basename(path).split("__", 1)[0],
                "run": r.get("run") or os.path.basename(path).rsplit(".", 1)[0].split("__")[-1],
                "epsilon": epsilon,
                "gap": round(float(ce.get("gap", 0) or 0), 4),
                "exposure_mean": round(float(ce.get("exposure_mean", 0) or 0), 2),
                "exposure_p95": round(float(ce.get("exposure_p95", 0) or 0), 2),
                "exposure_max": round(float(ce.get("exposure_max", 0) or 0), 2),
                "wbc_auc": float(wbc.get("wbc_auc", 0) or 0),
                "member_recall": member_recall.get("exact_secret_recall_rate"),
                "reference_recall": reference_recall.get("exact_secret_recall_rate"),
                "perplexity": perplexity.get("perplexity"),
            }
        )

    # Print table
    print(
        f"{'Model':<15} {'Run':<16} {'ε':<6} {'Gap':>8} {'Exp Mean':>9} "
        f"{'Exp P95':>8} {'WBC AUC':>8} {'Mem Rec':>8} {'Ref Rec':>8} {'PPL':>9}"
    )
    print("-" * 103)
    for r in sorted(rows, key=lambda x: (str(x["model"]), str(x["run"]), str(x["epsilon"]))):
        member_recall = "-" if r["member_recall"] is None else f"{float(r['member_recall']):.4f}"
        reference_recall = "-" if r["reference_recall"] is None else f"{float(r['reference_recall']):.4f}"
        perplexity = "-" if r["perplexity"] is None else f"{float(r['perplexity']):.2f}"
        print(
            f"{r['model']:<15} {str(r['run']):<16} {str(r['epsilon']):<6} {r['gap']:>+8.4f} "
            f"{r['exposure_mean']:>9.2f} {r['exposure_p95']:>8.2f} {r['wbc_auc']:>8.4f} "
            f"{member_recall:>8} {reference_recall:>8} {perplexity:>9}"
        )
    if not rows:
        print("(no result JSONs found)")
        print(f"Tried index: {args.index or os.path.join(SCRIPT_DIR, 'index.json')}")
        print(f"Tried glob:  {args.glob or os.path.join(SCRIPT_DIR, '*.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

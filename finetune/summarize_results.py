"""
Summarize all experiment results from results/ directory.

Usage:
    python summarize_results.py                    # all results
    python summarize_results.py --model gemma-2-2b-it   # one model only
    python summarize_results.py --pilot            # include pilot runs
"""

import os
import json
import glob
import argparse
from datetime import datetime


# ── Loader ────────────────────────────────────────────────────────────────────

def load_all_results(results_dir="results", model_filter=None, include_pilot=False):
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
            r["_file"] = os.path.basename(path)

        # Filter out pilot runs unless explicitly requested
        if r.get("pilot") and not include_pilot:
            continue

        # Filter by model if specified
        if model_filter and r.get("model_key") != model_filter:
            continue

        results.append(r)
    return results


# ── Summary sections ──────────────────────────────────────────────────────────

def print_baseline_summary(results):
    baselines = [r for r in results if r.get("run_type") == "baseline"]
    if not baselines:
        return

    print("\n📊 BASELINE LoRA")
    print("=" * 70)
    print(f"  {'Model':<20} {'Train Loss':>12} {'Runtime (hrs)':>14} {'Trainable%':>11}")
    print(f"  {'-'*60}")
    for r in sorted(baselines, key=lambda x: x["model_key"]):
        pct = 100 * r["trainable_params"] / r["total_params"]
        print(f"  {r['model_key']:<20} "
              f"{r['train_loss']:>12.4f} "
              f"{r['train_runtime_hrs']:>14.2f} "
              f"{pct:>10.2f}%")


def print_dp_summary(results):
    dp = [r for r in results if r.get("run_type") == "dp_lora"]
    if not dp:
        return

    print("\n🔒 DP-LoRA Results")
    print("=" * 85)
    print(f"  {'Model':<20} {'Target ε':>9} {'Final ε':>9} "
          f"{'Best Val':>10} {'Noise':>8} {'Hrs':>6}")
    print(f"  {'-'*65}")
    for r in sorted(dp, key=lambda x: (x["model_key"], x["target_epsilon"])):
        print(f"  {r['model_key']:<20} "
              f"{r['target_epsilon']:>9.1f} "
              f"{r['final_epsilon']:>9.4f} "
              f"{r['best_val_loss']:>10.4f} "
              f"{r['noise_multiplier']:>8.4f} "
              f"{r['train_runtime_hrs']:>6.2f}")


def print_tradeoff_table(results):
    print("\n📈 Privacy-Utility Tradeoff")
    print("=" * 55)

    model_keys = sorted(set(r["model_key"] for r in results))
    for model_key in model_keys:
        print(f"\n  {model_key}")
        print(f"  {'Run':<22} {'ε':>8} {'Val Loss':>10} {'Canary Gap':>12}")
        print(f"  {'-'*55}")

        model_results = [r for r in results if r["model_key"] == model_key]

        # Baseline
        for r in model_results:
            if r["run_type"] == "baseline":
                # Canary gap for baseline
                c_losses = r.get("canary_losses", [])
                ref_losses = r.get("reference_losses", [])
                gap = ""
                if c_losses and ref_losses:
                    avg_c   = sum(x["loss"] for x in c_losses) / len(c_losses)
                    avg_ref = sum(x["loss"] for x in ref_losses) / len(ref_losses)
                    gap = f"{avg_ref - avg_c:>+.4f}"
                val = r.get("train_loss", "N/A")
                print(f"  {'Baseline (no DP)':<22} {'N/A':>8} {val:>10} {gap:>12}")

        # DP sorted by epsilon
        dp_runs = [r for r in model_results if r["run_type"] == "dp_lora"]
        for r in sorted(dp_runs, key=lambda x: x["target_epsilon"]):
            # Get canary gap from last epoch
            gap = ""
            last_epoch = r.get("epoch_results", [{}])[-1]
            avg_c   = last_epoch.get("avg_canary_loss", 0)
            avg_ref = last_epoch.get("avg_reference_loss", 0)
            if avg_c and avg_ref:
                gap = f"{avg_ref - avg_c:>+.4f}"
            print(f"  DP ε={r['target_epsilon']:<22} "
                  f"{r['final_epsilon']:>8.3f} "
                  f"{r['best_val_loss']:>10.4f} "
                  f"{gap:>12}")


def print_canary_tracking(results):
    """Show how canary loss evolved per epoch — key for privacy paper."""
    dp = [r for r in results if r.get("run_type") == "dp_lora"]
    if not dp:
        return

    print("\n🕵️  Canary Loss Tracking (per epoch)")
    print("=" * 70)
    print("  Canary gap = ref_loss - member_loss")
    print("  Larger gap = model memorizes members more than non-members")
    print()

    for r in sorted(dp, key=lambda x: (x["model_key"], x["target_epsilon"])):
        epochs = r.get("epoch_results", [])
        if not epochs:
            continue
        print(f"  {r['model_key']}  ε={r['target_epsilon']}")
        print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
              f"{'ε spent':>9} {'Member':>10} {'Ref':>10} {'Gap':>8}")
        print(f"  {'-'*68}")
        for ep in epochs:
            avg_c   = ep.get("avg_canary_loss", 0)
            avg_ref = ep.get("avg_reference_loss", 0)
            gap     = avg_ref - avg_c if (avg_c and avg_ref) else 0
            print(f"  {ep['epoch']:>6} "
                  f"{ep['train_loss']:>12.4f} "
                  f"{ep['val_loss']:>10.4f} "
                  f"{ep['epsilon_spent']:>9.4f} "
                  f"{avg_c:>10.4f} "
                  f"{avg_ref:>10.4f} "
                  f"{gap:>+8.4f}")
        print()


def print_step_loss_summary(results):
    """Quick check that loss is going down — sanity check."""
    print("\n📉 Loss Curves (first / mid / last step)")
    print("=" * 60)

    for r in results:
        steps = r.get("step_losses", [])
        if len(steps) < 3:
            continue

        first = steps[0]["loss"]
        mid   = steps[len(steps)//2]["loss"]
        last  = steps[-1]["loss"]
        trend = "✅ going down" if last < first else "❌ not improving"

        tag = f"{r['model_key']} {r['run_type']}"
        if r["run_type"] == "dp_lora":
            tag += f" ε={r['target_epsilon']}"

        print(f"  {tag:<35} {first:.4f} → {mid:.4f} → {last:.4f}  {trend}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default=None,
                        help="Filter to one model key")
    parser.add_argument("--pilot",       action="store_true",
                        help="Include pilot runs in summary")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results = load_all_results(
        results_dir    = args.results_dir,
        model_filter   = args.model,
        include_pilot  = args.pilot,
    )

    if not results:
        print("No results found — have you run any experiments yet?")
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results   : {len(results)} runs  "
          f"({sum(1 for r in results if r['run_type']=='baseline')} baseline, "
          f"{sum(1 for r in results if r['run_type']=='dp_lora')} DP)")
    print("=" * 70)

    print_baseline_summary(results)
    print_dp_summary(results)
    print_tradeoff_table(results)
    print_canary_tracking(results)
    print_step_loss_summary(results)

    print("\n" + "=" * 70)
    print("Files in results/:")
    for r in results:
        tag = f"{r['model_key']} {r['run_type']}"
        if r["run_type"] == "dp_lora":
            tag += f" ε={r['target_epsilon']}"
        print(f"  {r['_file']:<55} {tag}")
    print("=" * 70)


if __name__ == "__main__":
    main()

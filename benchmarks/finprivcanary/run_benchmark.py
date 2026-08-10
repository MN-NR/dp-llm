import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.finprivcanary.runner import BenchmarkRunner


def main(model_name, checkpoint_dir, device, member_canary_path, ref_canary_path, out_dir, registry_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir if checkpoint_dir else model_name)
    model.to(device)

    runner = BenchmarkRunner(model, tokenizer, device=device)

    os.makedirs(out_dir, exist_ok=True)

    print("Running canary exposure...")
    exposure = runner.run_canary_exposure(member_canary_path, ref_canary_path, max_samples=200)
    runner.save_results(os.path.join(out_dir, "canary_exposure.json"), exposure)

    if registry_path:
        print("Running secret-token canary evaluation...")
        member_secret = runner.evaluate_canaries_during_training(
            member_canary_path,
            max_samples=200,
            registry_path=registry_path,
        )
        ref_secret = runner.evaluate_canaries_during_training(
            ref_canary_path,
            max_samples=200,
            registry_path=registry_path,
        )
        runner.save_results(
            os.path.join(out_dir, "secret_canary_eval.json"),
            {
                "member": member_secret.get("secret_evaluation"),
                "reference": ref_secret.get("secret_evaluation"),
            },
        )

    print("Running WBC attack (requires internet/git/requirements)...")
    wbc_out = runner.run_wbc(member_canary_path, ref_canary_path, os.path.join(out_dir, "wbc"), model_name)
    print("WBC output path:", wbc_out)

    print("Done. Results in", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--member_canary_path", type=str, required=True)
    parser.add_argument("--ref_canary_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="benchmarks/finprivcanary/results")
    parser.add_argument("--registry_path", type=str, default=None)
    args = parser.parse_args()
    main(
        args.model_name,
        args.checkpoint_dir,
        args.device,
        args.member_canary_path,
        args.ref_canary_path,
        args.out_dir,
        args.registry_path,
    )

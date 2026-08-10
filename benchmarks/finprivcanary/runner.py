import os
import json
import torch
from .canary_exposure import CanaryExposureEvaluator
from .wbc_wrapper import WBCAttackRunner
from .groupA_metrics import GroupAUtility
from benchmarks.synthetic_canary_audit import SyntheticCanaryAudit
from .crash_safe_checkpoint import CrashSafeCallback, load_canaries_for_checkpointing
from .canary_evaluator import compute_loss_statistics
from metrics.base import load_jsonl
from metrics.privacy.extraction import pii_extraction_rate
from metrics.privacy.inference import run_wbc_attack
from metrics.privacy.memorization import compute_canary_exposure_from_losses, exposure_by_category, exposure_by_repetition
from metrics.utility.lm import compute_perplexity, group_c_reconstruction


class BenchmarkRunner:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.exposure = CanaryExposureEvaluator(model, tokenizer, device=device)
        self.wbc = WBCAttackRunner(model=model, tokenizer=tokenizer, device=device)
        self.groupA = GroupAUtility()
        self.synthetic = SyntheticCanaryAudit(model, tokenizer, device=device)
        self.crash_callback = None

    def run_canary_exposure(self, member_path, reference_path, max_samples=200):
        member_losses = self.exposure.compute_canary_losses(member_path, max_samples)
        reference_losses = self.exposure.compute_canary_losses(reference_path, max_samples)
        exposure_score = self.exposure.compute_exposure(member_losses, reference_losses)

        result = {
            "member_losses": member_losses,
            "reference_losses": reference_losses,
            "exposure_score": exposure_score,
        }
        result.update(compute_canary_exposure_from_losses(member_losses, reference_losses))
        return result

    def run_wbc(self, member_path, reference_path, output_dir, model_name):
        return self.wbc.run_attack(member_path, reference_path, output_dir, model_name, device=self.device)

    def run_groupA_rouge(self, predictions, references):
        return self.groupA.compute_rouge(predictions, references)

    def run_group_a(self, eval_path, max_samples=500):
        return self.groupA.compute(self.model, self.tokenizer, eval_path, self.device, max_samples=max_samples)

    def run_group_c(self, eval_path, max_samples=200):
        return group_c_reconstruction(self.model, self.tokenizer, eval_path, self.device, max_samples=max_samples)

    def run_perplexity(self, eval_path, max_samples=500, group=None):
        return compute_perplexity(
            self.model,
            self.tokenizer,
            eval_path,
            self.device,
            max_samples=max_samples,
            group=group,
        )

    def run_pii_extraction(self, member_path, max_samples=500, max_new_tokens=30):
        member_canaries = load_jsonl(member_path, max_samples=max_samples)
        return pii_extraction_rate(
            self.model,
            self.tokenizer,
            member_canaries,
            self.device,
            max_new_tokens=max_new_tokens,
        )

    def run_repetition_study(self, member_path, reference_path, max_member_samples=500, max_reference_samples=1000):
        member_canaries = load_jsonl(member_path, max_samples=max_member_samples)
        reference_canaries = load_jsonl(reference_path, max_samples=max_reference_samples)
        member_losses = self.exposure.compute_canary_losses(member_path, max_member_samples)
        reference_losses = self.exposure.compute_canary_losses(reference_path, max_reference_samples)
        return {
            "exposure_by_category": exposure_by_category(member_losses, reference_losses),
            "exposure_by_repetition": exposure_by_repetition(member_losses, reference_losses),
            "num_member_canaries": len(member_canaries),
            "num_reference_canaries": len(reference_canaries),
        }

    def run_synthetic_audit(self, prompt_prefix, num_documents, canary_prefix, suffixes):
        return self.synthetic.run(prompt_prefix, num_documents, canary_prefix, suffixes)

    def save_results(self, out_path, results):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return out_path

    def setup_crash_safe_callback(self, privacy_engine, canary_path, reference_path, max_samples=200):
        """Setup crash-safe callback for training with canary evaluation."""
        member_canaries, reference_canaries = load_canaries_for_checkpointing(
            canary_path, reference_path, self.tokenizer, max_samples
        )
        
        self.crash_callback = CrashSafeCallback(
            privacy_engine=privacy_engine,
            delta=1e-5,
            member_canaries=member_canaries,
            reference_canaries=reference_canaries,
            save_every_steps=500
        )
        return self.crash_callback

    def evaluate_canaries_during_training(self, canary_path, max_samples=200, registry_path=None):
        raise NotImplementedError(
            "Deprecated: use `canary_eval.evaluate_canaries_full_sequence` and the paper metrics in `metrics/privacy/*`."
        )

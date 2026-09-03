import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from canarybench.tuning.config import DatasetConfig, PrivacyConfig, SearchSpace, TrainerConfig
from canarybench.tuning.preflight import PreflightReport
from canarybench.tuning.search import (
    launch_with_hooks,
    load_manifest,
    prepare_search,
    record_training_result,
    record_validation,
    search,
    select_search,
)
from canarybench.tuning.selection import (
    compute_u_val,
    normalize_validation_metrics,
    select_candidate,
    select_checkpoint,
    validate_privacy_budget,
    validate_selection_policy,
)


def metrics(ag=0.6, mnli=0.3, pii=0.9, invalid=0.1, repetition=0.2):
    return {
        "ag_news_accuracy": ag,
        "mnli_accuracy": mnli,
        "pii_unique_f1": pii,
        "pii_exact_match": 0.5,
        "invalid_rate": invalid,
        "repetition_fraction": repetition,
    }


class UtilitySelectionTests(unittest.TestCase):
    def test_u_val_calculation(self):
        self.assertAlmostEqual(compute_u_val(metrics()), 0.6)

    def test_checkpoint_selection(self):
        records = [
            {"epoch": 1, "checkpoint": "one", "validation": metrics(ag=0.3)},
            {"epoch": 2, "checkpoint": "two", "validation": metrics(ag=0.9)},
        ]
        self.assertEqual(select_checkpoint(records)["checkpoint"], "two")

    def test_tie_breaking_in_documented_order(self):
        base = metrics(ag=0.6, mnli=0.6, pii=0.6)
        records = [
            {"epoch": 5, "checkpoint": "repeat", "validation": dict(base, repetition_fraction=0.2)},
            {"epoch": 4, "checkpoint": "invalid", "validation": dict(base, invalid_rate=0.2, repetition_fraction=0.0)},
            {"epoch": 3, "checkpoint": "winner", "validation": dict(base, repetition_fraction=0.1)},
            {"epoch": 7, "checkpoint": "later", "validation": dict(base, repetition_fraction=0.1)},
        ]
        self.assertEqual(select_checkpoint(records)["checkpoint"], "winner")

    def test_privacy_budget_rejection(self):
        with self.assertRaisesRegex(ValueError, "exceeds target"):
            validate_privacy_budget(3.0, 3.01, tolerance=1e-3)
        validate_privacy_budget(3.0, 3.002, tolerance=1e-3)

    def test_test_split_and_test_metrics_rejected(self):
        with self.assertRaisesRegex(ValueError, "validation only"):
            validate_selection_policy("multitask_utility", "test")
        with self.assertRaisesRegex(ValueError, "not accepted"):
            normalize_validation_metrics(dict(metrics(), test_accuracy=1.0))

    def test_memorization_selection_metric_rejected(self):
        for name in ("memorization_gap", "wbc_auc", "exposure", "extraction"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_selection_policy(name, "validation")


class ManifestWorkflowTests(unittest.TestCase):
    def _dataset(self, root):
        data = Path(root) / "processed_v2"
        return DatasetConfig(
            root=str(data),
            train_path=str(data / "combined_train.jsonl"),
            validation_path=str(data / "combined_val.jsonl"),
            member_path=str(data / "member_canaries.jsonl"),
            reference_path=str(data / "attack_eval.jsonl"),
            registry_path=str(data / "canary_registry.json"),
        )

    def _prepare(self, root, privacy=None, policy="minimal"):
        privacy = privacy or PrivacyConfig(mode="dp", epsilon=3.0, delta=1e-5, accountant="rdp")
        backend = "repo_train_baseline" if privacy.mode == "none" else "repo_train_dp"
        return prepare_search(
            model="vault_gemma",
            privacy=privacy,
            search_space=SearchSpace(learning_rate=[2e-5, 1e-5], epochs=[20], global_batch_size=[128]),
            dataset=self._dataset(root),
            trainer=TrainerConfig(
                backend=backend, physical_batch_size=2,
                model_revision="abc123", tokenizer_revision="abc123",
            ),
            output_dir=Path(root) / "search",
            base_batch_size=2,
            lora={"r": 16, "alpha": 32, "target_modules": ["q_proj"]},
            search_policy=policy,
        )

    def test_candidate_serialization_deserialization(self):
        with tempfile.TemporaryDirectory() as root:
            prepared = self._prepare(root)
            loaded = load_manifest(Path(root) / "search" / "search_manifest.json")
            self.assertEqual(loaded["model"], prepared["model"])
            self.assertEqual(len(loaded["search"]["candidates"]), 2)
            self.assertEqual(loaded["search"]["candidates"][1]["status"], "deferred_approval_required")

    def test_incomplete_candidate_is_ignored(self):
        candidates = [
            {"id": "interrupted", "status": "running", "validation": []},
            {
                "id": "done", "status": "evaluated",
                "validation": [{"epoch": 1, "checkpoint": "ok", "validation": metrics()}],
            },
        ]
        self.assertEqual(select_candidate(candidates)["id"], "done")

    def test_non_dp_candidate_handling(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._prepare(root, PrivacyConfig(mode="none"))
            self.assertEqual(manifest["privacy"]["mode"], "none")
            self.assertNotIn("epsilon", {k for k, v in manifest["privacy"].items() if v is not None})
            self.assertEqual(manifest["trainer"]["backend"], "repo_train_baseline")

    def test_preflight_failure_prevents_candidate_launch(self):
        trainer = Mock(return_value={"should": "not run"})
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            launch_with_hooks(
                {},
                preflight=lambda: PreflightReport(False, errors=["EOS is masked"]),
                trainer=trainer,
            )
        trainer.assert_not_called()

    def test_end_to_end_record_and_select_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._prepare(root)
            path = Path(root) / "search" / "search_manifest.json"
            candidate_id = manifest["search"]["candidates"][0]["id"]
            record_training_result(path, candidate_id, {
                "final_epsilon": 2.99,
                "noise_multiplier": 1.23,
                "sample_rate": 0.01,
                "optimizer_steps": 100,
            })
            record_validation(path, candidate_id, epoch=1, checkpoint="epoch_01", metrics=metrics(ag=0.4))
            record_validation(path, candidate_id, epoch=2, checkpoint="epoch_02", metrics=metrics(ag=0.8))
            selected = select_search(path)
            self.assertEqual(selected["selected"]["epoch"], 2)
            self.assertFalse(selected["search"]["test_used"])
            self.assertFalse(selected["search"]["memorization_used_for_selection"])
            self.assertEqual(selected["audit"]["status"], "not_run")

    def test_public_api_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self._dataset(root)
            result = search(
                model="vault_gemma",
                privacy={"mode": "none"},
                search_space={"learning_rate": [2e-5], "epochs": [20], "global_batch_size": [128]},
                dataset=dataset.to_dict(),
                trainer={
                    "backend": "repo_train_baseline", "world_size": 1,
                    "physical_batch_size": 2, "model_revision": "abc",
                    "tokenizer_revision": "abc", "python_executable": "/usr/bin/python3",
                },
                output_dir=Path(root) / "api_search",
                base_batch_size=2,
                lora={"r": 16},
            )
            self.assertEqual(result["status"], "prepared")


if __name__ == "__main__":
    unittest.main()

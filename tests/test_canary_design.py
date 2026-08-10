import unittest
from unittest.mock import patch

from instruction_recall_dataset.build_instruction_dataset import validate_canary_isolation
from metrics.privacy.memorization import compute_canary_losses


class CanaryIsolationTests(unittest.TestCase):
    def setUp(self):
        self.registry = [
            {
                "source_text": "Contact Alice at alice@example.com",
                "secret": "alice@example.com",
            }
        ]

    def test_disjoint_ordinary_records_pass(self):
        records = [
            {
                "input": "Instruction: detect PII\n\nText: Contact Bob at bob@example.com",
                "target": "bob@example.com",
            }
        ]
        validate_canary_isolation(records, self.registry)

    def test_reserved_source_row_fails(self):
        records = [
            {
                "input": "Instruction: detect PII\n\nText: Contact Alice at alice@example.com",
                "target": "different@example.com",
            }
        ]
        with self.assertRaisesRegex(ValueError, "1 reserved source rows"):
            validate_canary_isolation(records, self.registry)

    def test_reserved_secret_in_another_row_fails(self):
        records = [
            {
                "input": "Instruction: detect PII\n\nText: Alternate context",
                "target": "alice@example.com",
            }
        ]
        with self.assertRaisesRegex(ValueError, "1 reserved secrets"):
            validate_canary_isolation(records, self.registry)


class CanaryLossTests(unittest.TestCase):
    @patch("metrics.privacy.memorization.compute_sequence_loss", return_value=1.25)
    def test_prompt_is_masked_by_default(self, compute_loss):
        records = [{"id": "canary-1", "input": "prefix", "target": "secret"}]

        losses = compute_canary_losses(
            model=object(),
            tokenizer=object(),
            canaries=records,
            device="cpu",
        )

        self.assertEqual(losses[0]["loss"], 1.25)
        self.assertTrue(compute_loss.call_args.kwargs["mask_prompt"])


if __name__ == "__main__":
    unittest.main()

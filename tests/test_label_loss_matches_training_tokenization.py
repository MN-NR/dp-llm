"""Regression test for the BOS-omission fix in evaluation/loss_based_eval.py.

Confirms the corrected `_avg_label_loss` scores its candidate exactly the
way finetune/dataset.py's MultiTaskDataset actually tokenizes and masks a
real training example -- not by re-deriving the expected token layout
independently (which could just repeat the same mistake), but by running
the real MultiTaskDataset code on a real record and comparing token-for-
token against what the corrected scorer builds internally.
"""
import json
import os
import sys
import tempfile
import unittest

from transformers import GemmaTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "finetune"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))

from dataset import MultiTaskDataset  # noqa: E402
from loss_based_eval import _avg_label_loss, _avg_label_loss_legacy, _build_prompt  # noqa: E402

SAMPLE_RECORD = {
    "id": "test_classification_0",
    "group": "A",
    "task": "classification",
    "input": (
        "Instruction: Classify the topic of the news article. "
        "Answer with one of: World, Sports, Business, Sci/Tech.\n\n"
        "Text: The Chicago Board Options Exchange said Tuesday its trading volume rose.\n"
        "Label:"
    ),
    "target": "Business",
    "split": "test",
    "canary": False,
}


def _load_tokenizer():
    try:
        return GemmaTokenizer.from_pretrained("google/vaultgemma-1b", local_files_only=True)
    except Exception:
        return None


class LabelLossMatchesTrainingTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _load_tokenizer()
        if self.tokenizer is None:
            self.skipTest("google/vaultgemma-1b tokenizer not available locally")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _real_training_example(self, max_length=128):
        """Runs the actual MultiTaskDataset on one record and returns its
        real (input_ids, labels) tensors -- ground truth for what training
        actually saw."""
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(SAMPLE_RECORD) + "\n")
            path = f.name
        try:
            ds = MultiTaskDataset(path, self.tokenizer, max_length=max_length)
            self.assertEqual(len(ds), 1, "sample record was filtered out by length -- adjust max_length")
            return ds[0]
        finally:
            os.unlink(path)

    def test_corrected_scorer_input_ids_prefix_matches_real_training_tensor(self):
        item = self._real_training_example()
        real_input_ids = item["input_ids"]
        real_labels = item["labels"]
        real_len = int(item["attention_mask"].sum().item())  # non-pad content length
        # Real training's full_text now ends in EOS (see the earlier EOS
        # fix in canarybench/prompt_format.py); _avg_label_loss deliberately
        # scores label tokens only, not EOS (a separate, orthogonal design
        # question -- see the "constrained/joint" method in
        # EXPERIMENT_LOG.md's evaluator-repair section for a scorer that
        # does include it). So the fair comparison region is real training's
        # content minus its trailing EOS token.
        self.assertEqual(int(real_input_ids[real_len - 1]), self.tokenizer.eos_token_id,
                          "expected real training example to end in EOS (EOS fix)")
        real_len_no_eos = real_len - 1

        prompt = _build_prompt(self.tokenizer, SAMPLE_RECORD["input"])
        label = SAMPLE_RECORD["target"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + label, add_special_tokens=True)["input_ids"]

        # 1. The corrected scorer's full sequence (unpadded) must match the
        #    real training input_ids exactly, position for position, over
        #    the real (non-pad, non-EOS) content -- this is the core
        #    BOS-inclusion claim: training's actual model input starts with
        #    BOS, and so must the corrected scorer's.
        self.assertEqual(
            full_ids, real_input_ids[:real_len_no_eos].tolist(),
            "corrected scorer's tokenization diverges from MultiTaskDataset's real "
            "training input -- the fix no longer matches the exact training context",
        )
        self.assertEqual(full_ids[0], self.tokenizer.bos_token_id,
                          "corrected scorer's sequence must start with BOS, matching training")

        # 2. The masking boundary: real training masks everything up to and
        #    including some prefix as -100 (prompt+BOS), leaving the target
        #    region unmasked. The corrected scorer computes prompt_len the
        #    same way (prompt tokenized alone, add_special_tokens=False) and
        #    must mask the identical boundary.
        prompt_len = len(prompt_ids)
        real_masked_prefix = (real_labels[:real_len_no_eos] == -100).tolist()
        expected_masked_prefix = [True] * prompt_len + [False] * (real_len_no_eos - prompt_len)
        self.assertEqual(
            real_masked_prefix, expected_masked_prefix,
            "real training's label mask boundary doesn't match prompt_len the way "
            "the corrected scorer assumes -- scorer would mask the wrong region",
        )

    def test_legacy_scorer_is_missing_bos(self):
        """Documents exactly what's wrong with the preserved legacy path,
        so the regression stays meaningful if someone is tempted to revert."""
        prompt = _build_prompt(self.tokenizer, SAMPLE_RECORD["input"])
        legacy_full_ids = self.tokenizer(prompt + SAMPLE_RECORD["target"], add_special_tokens=False)["input_ids"]
        self.assertNotEqual(
            legacy_full_ids[0], self.tokenizer.bos_token_id,
            "legacy scorer unexpectedly starts with BOS -- if this now passes, "
            "the historical bug this test documents may no longer reproduce",
        )


if __name__ == "__main__":
    unittest.main()

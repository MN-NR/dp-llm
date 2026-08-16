import json
import os
import tempfile
import unittest

from transformers import AutoTokenizer

from canarybench.prompt_format import (
    RESPONSE_PREFIX,
    build_supervised_text,
    legacy_hardcoded_supervised_text,
    tokenizer_uses_chat_template,
)

# Local HF hub cache repo ids -- these load from disk (local_files_only=True),
# no network required, matching the base models used by finetune/config.py's
# MODELS registry.
MODEL_TOKENIZERS = {
    "vault_gemma": "google/vaultgemma-1b",
    "gemma3_1b": "google/gemma-3-1b-it",
    "gemma_2b_it": "google/gemma-2-2b-it",
    "llama3_2_1b_it": "meta-llama/Llama-3.2-1B-Instruct",
}

SAMPLE_RECORD = {
    "id": "member_canary_example",
    "input": "Instruction: recall the secret.\n\nPrefix: My account ID is",
    "target": " ACCT-48213",
    "task": "pii_recall",
}


def _load_tokenizer(repo_id):
    try:
        return AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        return None


class SingleImplementationTests(unittest.TestCase):
    """Regression guard: training and evaluation must import the SAME
    formatter, not keep separate copies."""

    def test_metrics_base_reexports_canonical_formatter(self):
        import metrics.base as metrics_base
        import canarybench.prompt_format as canonical

        self.assertIs(metrics_base.build_supervised_text, canonical.build_supervised_text)
        self.assertIs(metrics_base.tokenizer_uses_chat_template, canonical.tokenizer_uses_chat_template)
        self.assertEqual(metrics_base.RESPONSE_PREFIX, canonical.RESPONSE_PREFIX)

    def test_multitask_dataset_uses_canonical_formatter(self):
        from finetune.dataset import MultiTaskDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "records.jsonl")
            record_with_group = dict(SAMPLE_RECORD, group="C")
            with open(path, "w") as f:
                f.write(json.dumps(record_with_group) + "\n")

            class _StubTokenizer:
                chat_template = None
                pad_token_id = 0

                def __call__(self, text, **kwargs):
                    return {"input_ids": list(range(len(text.split())))}

            ds = MultiTaskDataset(path, _StubTokenizer(), max_length=4096)
            full_text, prompt_only = ds._format_record(record_with_group)
            expected_prompt_only, expected_full_text = build_supervised_text(_StubTokenizer(), record_with_group)
            self.assertEqual(prompt_only, expected_prompt_only)
            self.assertEqual(full_text, expected_full_text)


class PromptAndTargetSpanTests(unittest.TestCase):
    """Demonstrates the actual formatted prompt and target span for each
    model family, per canarybench spec item 8."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {name: _load_tokenizer(repo) for name, repo in MODEL_TOKENIZERS.items()}

    def test_canonical_prompt_and_target_span_per_model(self):
        for name, tok in self.tokenizers.items():
            with self.subTest(model=name):
                if tok is None:
                    self.skipTest(f"{name} tokenizer not available locally")

                prompt_only, full_text = build_supervised_text(tok, SAMPLE_RECORD)
                self.assertTrue(full_text.startswith(prompt_only))

                prompt_ids = tok(prompt_only, add_special_tokens=False)["input_ids"]
                full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
                self.assertLessEqual(len(prompt_ids), len(full_ids))

                target_token_ids = full_ids[len(prompt_ids):]
                decoded_target_span = tok.decode(target_token_ids, skip_special_tokens=True)
                self.assertIn(SAMPLE_RECORD["target"].strip(), decoded_target_span)

                print(f"\n--- {name} ({MODEL_TOKENIZERS[name]}) ---")
                print("has_chat_template:", tokenizer_uses_chat_template(tok))
                print("prompt_only:", repr(prompt_only))
                print("full_text:  ", repr(full_text))
                print("target span (decoded from prompt_len: tokens):", repr(decoded_target_span))

    def test_vault_gemma_has_no_chat_template(self):
        """VaultGemma is a base (non-instruction-tuned) model: it has no
        chat_template, so canonical formatting falls back to the same
        "### Response:" template as the legacy path. This means the
        canary_eval.py prompt-formatting bugfix is a NO-OP for VaultGemma
        specifically -- it only changes behavior for chat-template models
        (Gemma-it, Llama-Instruct)."""
        tok = self.tokenizers["vault_gemma"]
        if tok is None:
            self.skipTest("vault_gemma tokenizer not available locally")
        self.assertFalse(tokenizer_uses_chat_template(tok))
        canonical = build_supervised_text(tok, SAMPLE_RECORD)
        legacy = legacy_hardcoded_supervised_text(SAMPLE_RECORD)
        self.assertEqual(canonical, legacy)

    def test_chat_template_models_diverge_from_legacy_hardcoded(self):
        """For Gemma-it/Llama-Instruct, the legacy hardcoded "### Response:"
        template is NOT what the model was trained on -- this is the bug
        that canary_eval.py etc. had before this fix."""
        for name in ("gemma3_1b", "gemma_2b_it", "llama3_2_1b_it"):
            tok = self.tokenizers[name]
            with self.subTest(model=name):
                if tok is None:
                    self.skipTest(f"{name} tokenizer not available locally")
                self.assertTrue(tokenizer_uses_chat_template(tok))
                canonical_prompt, _ = build_supervised_text(tok, SAMPLE_RECORD)
                legacy_prompt, _ = legacy_hardcoded_supervised_text(SAMPLE_RECORD)
                self.assertNotEqual(canonical_prompt, legacy_prompt)
                self.assertNotIn(RESPONSE_PREFIX, canonical_prompt)
                self.assertIn(RESPONSE_PREFIX, legacy_prompt)

    def test_lm_task_records_have_no_prompt(self):
        lm_record = {"task": "lm", "input": "", "target": "raw language modeling text"}
        for name, tok in self.tokenizers.items():
            with self.subTest(model=name):
                if tok is None:
                    self.skipTest(f"{name} tokenizer not available locally")
                prompt_only, full_text = build_supervised_text(tok, lm_record)
                self.assertEqual(prompt_only, "")
                self.assertEqual(full_text, lm_record["target"])


if __name__ == "__main__":
    unittest.main()

from metrics.base import load_jsonl, save_json
from metrics.privacy.inference import WindowBasedClassifierAttack


class WBCAttackRunner:
    def __init__(self, model=None, tokenizer=None, device="cuda", **_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def clone_repo(self):
        return None

    def install_requirements(self):
        return None

    def run_attack(
        self,
        member_file,
        reference_file,
        output_dir,
        model_name=None,
        device="cuda",
        window_sizes=None,
    ):
        if self.model is None or self.tokenizer is None:
            raise ValueError("WBCAttackRunner now requires model and tokenizer for local WBC evaluation.")
        member_records = load_jsonl(member_file)
        reference_records = load_jsonl(reference_file)
        attack = WindowBasedClassifierAttack(self.model, self.tokenizer, device=device)
        results = {}
        for window_size in window_sizes or [32, 64, 128]:
            results[f"window_{window_size}"] = attack.run(
                member_records,
                reference_records,
                window_lengths=[window_size],
            )
        return save_json(f"{output_dir}/wbc_results.json", results)

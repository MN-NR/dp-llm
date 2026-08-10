from typing import Any, Dict, List, Sequence

from metrics.base import generate_completion


class SyntheticCanaryAudit:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_documents(
        self,
        prompt_prefix: str,
        num_documents: int = 1000,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> List[str]:
        documents: List[str] = []
        for index in range(num_documents):
            prompt = prompt_prefix.replace("{i}", str(index))
            documents.append(
                generate_completion(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    device=self.device,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
            )
        return documents

    def detect_canary_echo(
        self,
        generated_docs: Sequence[str],
        canary_prefix: str,
        rare_suffix_candidates: Sequence[str],
    ) -> float:
        if not generated_docs:
            return 0.0
        hits = 0
        for document in generated_docs:
            if canary_prefix in document and any(suffix in document for suffix in rare_suffix_candidates):
                hits += 1
        return float(hits / len(generated_docs))

    def run(
        self,
        prompt_prefix: str,
        num_documents: int,
        canary_prefix: str,
        rare_suffix_candidates: Sequence[str],
    ) -> Dict[str, Any]:
        documents = self.generate_documents(prompt_prefix=prompt_prefix, num_documents=num_documents)
        return {
            "synthetic_docs": documents,
            "canary_echo_rate": self.detect_canary_echo(documents, canary_prefix, rare_suffix_candidates),
        }


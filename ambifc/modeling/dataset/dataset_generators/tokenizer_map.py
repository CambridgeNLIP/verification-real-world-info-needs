from typing import Dict

from transformers import AutoTokenizer


class TokenizeMap:
    def __init__(self, tokenizer: AutoTokenizer, evidence_key: str = 'evidence', claim_key: str = 'claim'):
        self.tokenizer: AutoTokenizer = tokenizer
        self.evidence_key: str = evidence_key
        self.claim_key: str = claim_key

    def map(self, sample: Dict):
        claim: str = sample[self.claim_key]
        evidence: str = sample[self.evidence_key]

        result: Dict = self.tokenizer(
            claim, evidence, add_special_tokens=True, return_tensors='pt'
        )

        return {
            k: result[k][0] for k in result
        }

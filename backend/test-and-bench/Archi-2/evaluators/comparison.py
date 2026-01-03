import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from core.metrics import ComparisonResult


class ComparisonEvaluator:
    """
    AI vs Human answer comparison.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, ai_answer: str, human_answer: str) -> ComparisonResult:
        ai_emb = self.model.encode(ai_answer)
        human_emb = self.model.encode(human_answer)

        semantic = np.dot(ai_emb, human_emb) / (
            np.linalg.norm(ai_emb) * np.linalg.norm(human_emb) + 1e-8
        )

        ai_tokens = set(ai_answer.lower().split())
        human_tokens = set(human_answer.lower().split())
        lexical = len(ai_tokens & human_tokens) / len(ai_tokens | human_tokens)

        structural = SequenceMatcher(None, ai_answer, human_answer).ratio()
        length_ratio = len(ai_tokens) / len(human_tokens) if human_tokens else 1.0

        agreement = 0.5 * semantic + 0.3 * lexical + 0.2 * structural

        recommendation = (
            "ACCEPT" if agreement > 0.85 else
            "REVIEW" if agreement > 0.6 else
            "REJECT"
        )

        return ComparisonResult(
            ai_answer=ai_answer,
            human_answer=human_answer,
            semantic_similarity=float(min(1, semantic)),
            lexical_similarity=float(min(1, lexical)),
            structural_similarity=float(min(1, structural)),
            length_ratio=float(min(2, length_ratio)),
            agreement_score=float(min(1, agreement)),
            recommendation=recommendation
        )

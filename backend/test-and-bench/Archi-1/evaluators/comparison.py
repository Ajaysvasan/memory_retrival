import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from core.metrics import ComparisonResult


class ComparisonEvaluator:
    """
    Compares AI-generated answer with a human-written answer.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, ai_answer: str, human_answer: str) -> ComparisonResult:
        ai_emb = self.model.encode(ai_answer)
        human_emb = self.model.encode(human_answer)

        semantic = float(
            np.dot(ai_emb, human_emb) /
            (np.linalg.norm(ai_emb) * np.linalg.norm(human_emb) + 1e-8)
        )

        ai_tokens = set(ai_answer.lower().split())
        human_tokens = set(human_answer.lower().split())
        lexical = len(ai_tokens & human_tokens) / len(ai_tokens | human_tokens)

        structural = SequenceMatcher(None, ai_answer, human_answer).ratio()

        ai_len = len(ai_answer.split())
        human_len = len(human_answer.split())
        length_ratio = ai_len / human_len if human_len else 1.0

        agreement = 0.5 * semantic + 0.3 * lexical + 0.2 * structural

        if agreement > 0.9:
            recommendation = "ACCEPT"
        elif agreement > 0.75:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"

        return ComparisonResult(
            ai_answer=ai_answer,
            human_answer=human_answer,
            semantic_similarity=min(1.0, semantic),
            lexical_similarity=min(1.0, lexical),
            structural_similarity=min(1.0, structural),
            length_ratio=min(2.0, length_ratio),
            agreement_score=min(1.0, agreement),
            recommendation=recommendation,
        )

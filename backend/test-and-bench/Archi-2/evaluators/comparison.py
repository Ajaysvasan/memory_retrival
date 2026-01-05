import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from core.metrics import ComparisonResult


class ComparisonEvaluator:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, ai, human):
        a = self.model.encode(ai)
        h = self.model.encode(human)

        semantic = np.dot(a, h) / (np.linalg.norm(a) * np.linalg.norm(h) + 1e-8)
        structural = SequenceMatcher(None, ai, human).ratio()
        agreement = 0.7 * semantic + 0.3 * structural

        return ComparisonResult(
            ai, human, semantic, semantic, structural, 1.0,
            agreement,
            "ACCEPT" if agreement > 0.7 else "REVIEW"
        )

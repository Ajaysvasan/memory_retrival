import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from core.metrics import ComparisonResult

class ComparisonEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, ai, human):
        sim = float(np.dot(self.model.encode(ai), self.model.encode(human)) /
            (np.linalg.norm(self.model.encode(ai))*np.linalg.norm(self.model.encode(human))+1e-8))
        rec = "✓ ACCEPT" if sim > 0.85 else "⚠ REVIEW" if sim > 0.5 else "✗ REJECT"
        return ComparisonResult(ai, human, sim, sim, sim, 1.0, sim, rec)

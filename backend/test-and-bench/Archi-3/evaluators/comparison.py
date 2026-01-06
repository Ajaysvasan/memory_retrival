import numpy as np
from sentence_transformers import SentenceTransformer
from core.metrics import ComparisonResult

class ComparisonEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, ai, human):
        sim = float(np.dot(self.model.encode(ai), self.model.encode(human)) /
            (np.linalg.norm(self.model.encode(ai))*np.linalg.norm(self.model.encode(human))+1e-8))

        if sim > 0.85:
            rec = "ACCEPT"
        elif sim > 0.6:
            rec = "REVIEW"
        else:
            rec = "REJECT"

        return ComparisonResult(sim, rec)

import numpy as np
from sentence_transformers import SentenceTransformer
from core.metrics import ConsistencyScore

class ConsistencyEvaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, query, docs):
        q = self.model.encode(query)
        sims = [
            float(np.dot(q, self.model.encode(d.content)) /
            (np.linalg.norm(q)*np.linalg.norm(self.model.encode(d.content))+1e-8))
            for d in docs
        ]
        return ConsistencyScore(overall_consistency=float(np.mean(sims)))

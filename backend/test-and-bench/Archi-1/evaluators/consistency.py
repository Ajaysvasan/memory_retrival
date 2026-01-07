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
        semantic = np.mean(sims)
        lexical = np.mean([
            len(set(query.split()) & set(d.content.split())) /
            len(set(query.split()) | set(d.content.split()))
            for d in docs
        ])
        overall = 0.6*semantic + 0.4*lexical
        variation = np.std(sims)/(np.mean(sims)+1e-8)
        return ConsistencyScore(semantic, lexical, overall, variation)

import numpy as np
from sentence_transformers import SentenceTransformer
from core.metrics import ConsistencyScore

class ConsistencyEvaluator:
    def __init__(self, vector_db):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_db = vector_db

    def evaluate(self, query, docs):
        q = self.model.encode(query)
        sims = [
            float(np.dot(q, self.model.encode(d.content)) /
            (np.linalg.norm(q)*np.linalg.norm(self.model.encode(d.content))+1e-8))
            for d in docs
        ]
        semantic = np.mean(sims)
        lexical = np.mean([len(set(query.split()) & set(d.content.split())) /
                            len(set(query.split()) | set(d.content.split()))
                            for d in docs])
        vector_score = min(1.0, len(docs) / 10.0)
        overall = 0.4*semantic + 0.3*lexical + 0.3*vector_score
        variation = np.std(sims) / (np.mean(sims)+1e-8)

        return ConsistencyScore(semantic, lexical, vector_score, overall, variation)

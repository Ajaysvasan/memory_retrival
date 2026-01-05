import numpy as np
from sentence_transformers import SentenceTransformer
from core.metrics import ConsistencyScore


class ConsistencyEvaluator:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, query, docs, fusion_stats):
        q = self.model.encode(query)
        d = [self.model.encode(x.content) for x in docs]

        semantic = np.mean([
            np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-8)
            for e in d
        ])

        lexical = np.mean([
            len(set(query.split()) & set(x.content.split())) /
            len(set(query.split()) | set(x.content.split()))
            for x in docs
        ])

        fusion = 0.0
        if "attention_weights" in fusion_stats:
            a = np.array(fusion_stats["attention_weights"])
            fusion = 1 - (-np.sum(a * np.log(a + 1e-8)) / np.log(len(a)))

        overall = 0.4 * semantic + 0.3 * lexical + 0.3 * fusion
        variation = np.std(d) / (np.mean(d) + 1e-8)

        return ConsistencyScore(
            semantic, lexical, fusion, overall, float(variation)
        )

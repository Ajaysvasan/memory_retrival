import numpy as np
from sentence_transformers import SentenceTransformer
from core.metrics import ConsistencyScore
from core.retrieved_doc import RetrievedDoc
from typing import List


class ConsistencyEvaluator:
    """
    Evaluates how consistent retrieved documents are with respect to a query.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, query: str, retrieved_docs: List[RetrievedDoc]) -> ConsistencyScore:
        if not retrieved_docs:
            return ConsistencyScore(0.0, 0.0, 0.0, 0.0)

        query_emb = self.model.encode(query)
        doc_embs = [self.model.encode(doc.content) for doc in retrieved_docs]

        # Semantic consistency (cosine similarity)
        semantic_scores = [
            float(np.dot(query_emb, emb) /
                  (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
            for emb in doc_embs
        ]
        semantic_consistency = float(np.mean(semantic_scores))

        # Lexical consistency (Jaccard similarity)
        query_tokens = set(query.lower().split())
        lexical_scores = []
        for doc in retrieved_docs:
            doc_tokens = set(doc.content.lower().split())
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            lexical_scores.append(intersection / union if union > 0 else 0.0)

        lexical_consistency = float(np.mean(lexical_scores))

        overall_consistency = 0.6 * semantic_consistency + 0.4 * lexical_consistency
        variation_coefficient = float(
            np.std(semantic_scores) / (np.mean(semantic_scores) + 1e-8)
        )

        return ConsistencyScore(
            semantic_consistency=min(1.0, semantic_consistency),
            lexical_consistency=min(1.0, lexical_consistency),
            overall_consistency=min(1.0, overall_consistency),
            variation_coefficient=variation_coefficient,
        )

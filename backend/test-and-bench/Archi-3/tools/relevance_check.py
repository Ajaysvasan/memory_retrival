import numpy as np
from sentence_transformers import SentenceTransformer
from tools.base import BaseTool


class RelevanceCheckTool(BaseTool):
    """
    Validate whether retrieved documents are relevant to the query
    using semantic similarity.
    """

    def __init__(self, threshold: float = 0.3):
        super().__init__("check_relevance")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def execute(self, query: str, documents, **kwargs):
        if not documents:
            return {
                "is_relevant": False,
                "relevant_ratio": 0.0,
                "avg_relevance": 0.0,
                "relevance_scores": [],
            }

        query_emb = self.model.encode(query)
        scores = []

        for doc in documents:
            doc_emb = self.model.encode(doc.content)
            sim = float(
                np.dot(query_emb, doc_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8)
            )
            scores.append(sim)

        relevant_count = sum(1 for s in scores if s >= self.threshold)

        return {
            "is_relevant": relevant_count / len(scores) >= 0.5,
            "relevant_ratio": relevant_count / len(scores),
            "avg_relevance": float(np.mean(scores)),
            "relevance_scores": scores,
        }

from typing import List
from core.metrics import ComplexityMetrics


class ComplexityAnalyzer:
    """
    Time and space complexity analysis for FiD RAG.
    """

    @staticmethod
    def get_complexity_analysis(
        n_docs: int,
        k_retrieved: int = 10,
        embedding_dim: int = 384,
        hidden_dim: int = 768
    ) -> List[ComplexityMetrics]:

        return [
            ComplexityMetrics(
                "Semantic Indexing",
                "O(n*d)",
                "O(n*d)",
                f"{n_docs} documents embedded"
            ),
            ComplexityMetrics(
                "Hybrid Retrieval",
                "O(n*d + q log n)",
                "O(k)",
                f"Retrieve top-{k_retrieved}"
            ),
            ComplexityMetrics(
                "FiD Encoding",
                "O(k * Transformer)",
                "O(k*d)",
                "Encode query-document pairs"
            ),
            ComplexityMetrics(
                "Fusion & Attention",
                "O(k*d)",
                "O(k)",
                "Fuse representations"
            ),
        ]

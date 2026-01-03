from core.metrics import ComplexityMetrics
from typing import List


class ComplexityAnalyzer:
    """
    Provides theoretical complexity analysis of the RAG pipeline.
    """

    @staticmethod
    def get_complexity_analysis(n_docs: int, embedding_dim: int = 384) -> List[ComplexityMetrics]:
        return [
            ComplexityMetrics(
                operation="Semantic Indexing",
                time_complexity="O(n * d)",
                space_complexity="O(n * d)",
                description=f"Embedding {n_docs} documents"
            ),
            ComplexityMetrics(
                operation="BM25 Indexing",
                time_complexity="O(n * m)",
                space_complexity="O(n * m)",
                description="Sparse inverted index creation"
            ),
            ComplexityMetrics(
                operation="Retrieval",
                time_complexity="O(n * d)",
                space_complexity="O(d)",
                description="Similarity computation"
            ),
            ComplexityMetrics(
                operation="Cross-Encoder Reranking",
                time_complexity="O(k * d)",
                space_complexity="O(k * d)",
                description="Pairwise relevance scoring"
            ),
            ComplexityMetrics(
                operation="Evaluation",
                time_complexity="O(k²)",
                space_complexity="O(k)",
                description="Consistency & accuracy evaluation"
            ),
        ]

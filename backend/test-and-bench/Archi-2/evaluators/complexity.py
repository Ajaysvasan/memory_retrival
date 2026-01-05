from core.metrics import ComplexityMetrics


class ComplexityAnalyzer:

    def get_complexity_analysis(self, n_docs, k):
        return [
            ComplexityMetrics(
                "FiD RAG",
                "O(n*d + k*d)",
                "O(n*d)",
                "End-to-end FiD pipeline"
            )
        ]

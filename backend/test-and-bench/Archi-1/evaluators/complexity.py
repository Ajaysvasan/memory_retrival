from core.metrics import ComplexityMetrics

class ComplexityAnalyzer:
    @staticmethod
    def get_analysis(n_docs):
        return [
            ComplexityMetrics("Vector DB Index","O(n*d)","O(n*d)","Embedding"),
            ComplexityMetrics("Hybrid Search","O(k*d)","O(k)","Retrieval"),
            ComplexityMetrics("Cross Encoder","O(k*d)","O(k*d)","Reranking"),
        ]

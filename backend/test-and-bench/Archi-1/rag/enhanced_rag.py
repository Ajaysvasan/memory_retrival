import time
from retrievers.hybrid import HybridRetriever
from retrievers.reranker import CrossEncoderReranker
from evaluators.consistency import ConsistencyEvaluator
from evaluators.accuracy import AccuracyEvaluator
from evaluators.comparison import ComparisonEvaluator
from evaluators.complexity import ComplexityAnalyzer
from core.evaluation_result import EvaluationResult

class EnhancedRAGSystem:

    def __init__(self, alpha=0.5, rerank_top_k=5):
        self.retriever = HybridRetriever(alpha)
        self.reranker = CrossEncoderReranker()
        self.consistency = ConsistencyEvaluator()
        self.accuracy = AccuracyEvaluator()
        self.comparison = ComparisonEvaluator()
        self.complexity = ComplexityAnalyzer()
        self.rerank_top_k = rerank_top_k
        self.n_docs = 0

    def index(self, documents):
        self.n_docs = len(documents)
        self.retriever.index(documents)

    def retrieve(self, query, k):
        docs = self.retriever.retrieve(query, k)
        texts = [d.content for d in docs]
        ranked = self.reranker.rerank(query, texts, self.rerank_top_k)
        return [docs[i] for i, _ in ranked]

    def evaluate_full(self, query, ai_answer, human_answer, k=20):
        start = time.time()
        docs = self.retrieve(query, k)

        return EvaluationResult(
            query=query,
            retrieved_docs=docs,
            consistency=self.consistency.evaluate(query, docs),
            accuracy=self.accuracy.evaluate(ai_answer, human_answer),
            comparison=self.comparison.evaluate(ai_answer, human_answer),
            complexity=self.complexity.get_complexity_analysis(self.n_docs),
            total_time=time.time() - start
        )

import time
from retrievers.hybrid_retriever import HybridRetriever
from fid.answer_generator import AnswerGenerator
from fid.fusion_encoder import FusionEncoder
from evaluators.consistency import ConsistencyEvaluator
from evaluators.accuracy import AccuracyEvaluator
from evaluators.comparison import ComparisonEvaluator
from core.evaluation_result import EvaluationResult
from core.metrics import ComplexityMetrics

class FiDRAGwithVectorDB:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.answer_gen = AnswerGenerator()
        self.encoder = FusionEncoder()
        self.accuracy = AccuracyEvaluator()
        self.comparison = ComparisonEvaluator()

    def setup(self, docs):
        self.retriever.index(docs)
        self.consistency = ConsistencyEvaluator(self.retriever.vector_db)

    def evaluate(self, query, human):
        start = time.time()
        docs = self.retriever.search(query)
        answer, _ = self.answer_gen.generate(query, docs, self.encoder)
        consistency = self.consistency.evaluate(query, docs)
        accuracy = self.accuracy.evaluate(answer, human)
        comparison = self.comparison.evaluate(answer, human)

        return EvaluationResult(
            query, docs, answer, consistency, accuracy, comparison,
            [ComplexityMetrics("FiD RAG", "O(n*d)", "O(n*d)", "End-to-end")],
            time.time() - start,
            self.retriever.vector_db.get_stats()
        )

import time
from retrievers.hybrid import HybridRetriever
from fid.answer_generator import AnswerGeneratorFiD
from fid.fusion_encoder import FusionEncoder
from evaluators.consistency import ConsistencyEvaluator
from evaluators.accuracy import AccuracyEvaluator
from evaluators.comparison import ComparisonEvaluator
from evaluators.complexity import ComplexityAnalyzer
from core.evaluation_result import EvaluationResult


class FusionInDecoderRAG:

    def __init__(self, alpha=0.6, k=8):
        self.retriever = HybridRetriever(alpha)
        self.generator = AnswerGeneratorFiD()
        self.encoder = FusionEncoder()
        self.consistency = ConsistencyEvaluator()
        self.accuracy = AccuracyEvaluator()
        self.comparison = ComparisonEvaluator()
        self.complexity = ComplexityAnalyzer()
        self.k = k

    def index(self, docs):
        self.retriever.index(docs)
        self.n_docs = len(docs)

    def evaluate_full(self, query, human):
        start = time.time()
        docs = self.retriever.retrieve(query, self.k)
        ans, stats = self.generator.generate_answer(query, docs, self.encoder)

        return EvaluationResult(
            query,
            docs,
            ans,
            self.consistency.evaluate(query, docs, stats),
            self.accuracy.evaluate(ans, human),
            self.comparison.evaluate(ans, human),
            self.complexity.get_complexity_analysis(self.n_docs, self.k),
            time.time() - start,
            stats
        )

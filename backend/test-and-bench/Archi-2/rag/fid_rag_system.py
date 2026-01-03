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

    def __init__(self, retrieval_alpha=0.5, num_retrieved=10, use_fusion=True):
        self.retriever = HybridRetriever(retrieval_alpha)
        self.generator = AnswerGeneratorFiD()
        self.consistency = ConsistencyEvaluator()
        self.accuracy = AccuracyEvaluator()
        self.comparison = ComparisonEvaluator()
        self.complexity = ComplexityAnalyzer()

        self.num_retrieved = num_retrieved
        self.fusion_encoder = (
            FusionEncoder(num_documents=num_retrieved) if use_fusion else None
        )
        self.n_docs = 0

    def index(self, documents):
        self.n_docs = len(documents)
        self.retriever.index(documents)

    def evaluate_full(self, query, human_answer):
        start = time.time()

        docs = self.retriever.retrieve(query, self.num_retrieved)
        answer, fusion_stats = self.generator.generate_answer(
            query, docs, self.fusion_encoder
        )

        return EvaluationResult(
            query=query,
            retrieved_docs=docs,
            generated_answer=answer,
            consistency=self.consistency.evaluate(query, docs, fusion_stats),
            accuracy=self.accuracy.evaluate(answer, human_answer),
            comparison=self.comparison.evaluate(answer, human_answer),
            complexity=self.complexity.get_complexity_analysis(self.n_docs, self.num_retrieved),
            total_time=time.time() - start,
            fusion_mechanism_stats=fusion_stats,
        )

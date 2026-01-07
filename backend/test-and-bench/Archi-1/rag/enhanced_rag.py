import time
from retrievers.hybrid_retriever import HybridRetrieverWithVectorDB
from reranker.cross_encoder import CrossEncoderReranker
from evaluators.consistency import ConsistencyEvaluator
from evaluators.accuracy import AccuracyEvaluator
from evaluators.comparison import ComparisonEvaluator
from evaluators.complexity import ComplexityAnalyzer
from core.evaluation_result import EvaluationResult

class HybridRAGwithVectorDB:
    def __init__(self):
        self.retriever = HybridRetrieverWithVectorDB()
        self.reranker = CrossEncoderReranker()
        self.consistency = ConsistencyEvaluator()
        self.accuracy = AccuracyEvaluator()
        self.comparison = ComparisonEvaluator()

    def setup(self, docs):
        self.docs = docs
        self.retriever.index(docs)

    def evaluate(self, query, human):
        start = time.time()

        candidates = self.retriever.search(query, 10)
        texts = [d.content for d in candidates]
        reranked = self.reranker.rerank(query, texts, 5)

        final_docs = []
        for idx, score in reranked:
            d = candidates[idx]
            d.rerank_score = score
            final_docs.append(d)

        answer = " ".join(d.content for d in final_docs[:2])

        return EvaluationResult(
            query=query,
            retrieved_docs=final_docs,
            generated_answer=answer,
            consistency=self.consistency.evaluate(query, final_docs),
            accuracy=self.accuracy.evaluate(answer, human),
            comparison=self.comparison.evaluate(answer, human),
            complexity=ComplexityAnalyzer.get_analysis(len(self.docs)),
            total_time=time.time()-start,
            vector_db_stats=self.retriever.vector_db.get_stats()
        )

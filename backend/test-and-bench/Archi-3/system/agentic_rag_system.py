import time
from vectordb.inmemory import InMemoryVectorDB
from tools.vector_search import VectorDBSearchTool
from tools.bm25_search import BM25SearchTool
from tools.hybrid_search import HybridSearchTool
from tools.query_decompose import QueryDecompositionTool
from tools.answer_synthesis import AnswerSynthesisTool
from agent.orchestrator import AgentOrchestrator
from evaluators.consistency import ConsistencyEvaluator
from evaluators.comparison import ComparisonEvaluator
from core.evaluation_result import EvaluationResult
from tools.relevance_check import RelevanceCheckTool

class AgenticRAGSystem:
    def __init__(self, documents):
        self.db = InMemoryVectorDB()
        self.db.add_documents(documents)

        vector_tool = VectorDBSearchTool(self.db)
        bm25_tool = BM25SearchTool(documents)
        hybrid_tool = HybridSearchTool(vector_tool, bm25_tool)

        self.agent = AgentOrchestrator()
        self.agent.register(hybrid_tool)
        self.agent.register(RelevanceCheckTool())     
        self.agent.register(AnswerSynthesisTool())

        self.consistency = ConsistencyEvaluator()
        self.comparison = ComparisonEvaluator()


    def run(self, query, human):
        start = time.time()
        answer, docs = self.agent.run(query)

        return EvaluationResult(
            query=query,
            consistency=self.consistency.evaluate(query, docs),
            comparison=self.comparison.evaluate(answer, human),
            total_time=float(time.time() - start)
        )

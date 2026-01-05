import time
from tools.retrieval import HybridSearchTool
from tools.synthesis import AnswerSynthesisTool
from agent.orchestrator import AgentOrchestrator
from evaluators.consistency import ConsistencyEvaluator
from evaluators.comparison import ComparisonEvaluator
from core.evaluation_result import EvaluationResult

class AgenticRAGSystem:
    def __init__(self, documents):
        self.retriever = HybridSearchTool(documents)
        self.synth = AnswerSynthesisTool()
        self.agent = AgentOrchestrator(self.retriever, self.synth)
        self.consistency = ConsistencyEvaluator()
        self.comparison = ComparisonEvaluator()

    def run(self, query, human_answer):
        start = time.time()
        answer, docs = self.agent.run(query)

        return EvaluationResult(
            query=query,
            consistency=self.consistency.evaluate(query, docs),
            comparison=self.comparison.evaluate(answer, human_answer),
            total_time=float(time.time() - start)
        )

from typing import Optional

from config import Config
from core.contracts.intent import Intent
from core.contracts.phase_a_decision import PhaseADecision
from core.contracts.query import Query
from core.contracts.retrieval_plan import RetrievalPlan
from core.errors.refusal_reason import RefusalReason
from logger import Logger
from modules.intent.intent_classifier import IntentClassifier
from modules.planning.retrival_planner import RetrievalPlanner
from modules.rl.rl_agent import RLAgent


class PhaseAOrchestrator:
    """Orchestrates Phase A: Intent → Plan"""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        retrieval_planner: RetrievalPlanner,
        rl_agent: RLAgent,
    ):
        self.intent_classifier = intent_classifier
        self.retrieval_planner = retrieval_planner
        self.rl_agent = rl_agent
        self.logger = Logger().get_logger("PhaseA")

    def orchestrate(self, query: Query, context: dict) -> PhaseADecision:
        """Execute Phase A orchestration"""

        self.logger.info("=== Phase A: Intent Classification & Planning ===")

        intent = self.intent_classifier.classify(query)
        self.logger.info(
            f"Intent: {intent.intent_type.value} (confidence={intent.confidence:.2f})"
        )

        state_features = self.rl_agent.extract_state_features(context)
        rl_decisions = self.rl_agent.make_decisions(state_features)

        self.logger.debug(f"RL decisions: {rl_decisions}")

        plan = self.retrieval_planner.create_plan(query, intent, rl_decisions)
        self.logger.info(
            f"Plan created: {len(plan.topic_ids)} topics, {plan.max_chunks} chunks"
        )

        return PhaseADecision(
            intent=intent, plan=plan, should_proceed=True, refusal_reason=None
        )

from typing import Optional
from core.contracts.query import Query
from core.contracts.intent import Intent
from core.contracts.retrieval_plan import RetrievalPlan
from core.contracts.phase_a_decision import PhaseADecision
from core.errors.refusal_reason import RefusalReason
from modules.intent.intent_classifier import IntentClassifier
from modules.planning.retrival_planner import RetrievalPlanner
from modules.rl.rl_agent import RLAgent
from logger import Logger
from config import Config


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

        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        self.logger.info(
            f"Intent: {intent.intent_type.value} (confidence={intent.confidence:.2f})"
        )

        # Step 2: Check intent confidence
        if not intent.is_confident:
            return PhaseADecision(
                intent=intent,
                plan=None,
                should_proceed=False,
                refusal_reason=RefusalReason.UNCERTAIN_INTENT.to_message(),
            )

        # Step 3: Get RL decisions
        state_features = self.rl_agent.extract_state_features(context)
        rl_decisions = self.rl_agent.make_decisions(state_features)

        self.logger.debug(f"RL decisions: {rl_decisions}")

        # Step 4: Check if RL suggests refusal
        if rl_decisions.get("refuse", False):
            return PhaseADecision(
                intent=intent,
                plan=None,
                should_proceed=False,
                refusal_reason=RefusalReason.SYSTEM_ERROR.to_message(),
            )

        # Step 5: Create retrieval plan
        plan = self.retrieval_planner.create_plan(query, intent, rl_decisions)
        self.logger.info(
            f"Plan: {len(plan.topic_ids)} topics, {plan.max_chunks} chunks, cache={plan.use_cache}"
        )

        # Step 6: Return successful decision
        return PhaseADecision(
            intent=intent, plan=plan, should_proceed=True, refusal_reason=None
        )

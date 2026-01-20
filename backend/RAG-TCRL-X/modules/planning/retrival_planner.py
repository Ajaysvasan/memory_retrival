from typing import List, Set, FrozenSet
from datetime import datetime
from core.contracts.query import Query
from core.contracts.intent import Intent
from core.contracts.retrieval_plan import RetrievalPlan
from modules.intake.query_intake import QueryIntake
from logger import Logger
from config import Config


class RetrievalPlanner:
    """Generate retrieval execution plans"""

    def __init__(self, num_topics: int):
        self.num_topics = num_topics
        self.logger = Logger().get_logger("RetrievalPlanner")

    def create_plan(
        self, query: Query, intent: Intent, rl_decisions: dict
    ) -> RetrievalPlan:
        """Create retrieval plan based on query, intent, and RL decisions"""

        query_hash = QueryIntake.compute_query_hash(query)

        # Determine topics based on intent
        topic_ids = self._select_topics(
            intent, rl_decisions.get("expand_topics", False)
        )

        # Determine retrieval strategy
        use_cache = rl_decisions.get("use_cache", True)
        use_ann = rl_decisions.get("use_ann", True)
        expand_topics = rl_decisions.get("expand_topics", False)

        # Determine chunk count based on intent
        max_chunks = self._determine_chunk_count(intent)

        plan = RetrievalPlan(
            query_hash=query_hash,
            topic_ids=frozenset(topic_ids),
            max_chunks=max_chunks,
            use_cache=use_cache,
            use_ann=use_ann,
            expand_topics=expand_topics,
            timestamp=datetime.now(),
        )

        self.logger.debug(
            f"Created plan: topics={len(topic_ids)}, chunks={max_chunks}, cache={use_cache}"
        )
        return plan

    def _select_topics(self, intent: Intent, expand: bool) -> Set[int]:
        """Select topics based on intent"""
        from core.contracts.intent import IntentType

        # Base topic selection
        if intent.intent_type == IntentType.FACTUAL:
            base_count = 2
        elif intent.intent_type == IntentType.ANALYTICAL:
            base_count = 3
        elif intent.intent_type == IntentType.COMPARATIVE:
            base_count = 4
        elif intent.intent_type == IntentType.PROCEDURAL:
            base_count = 3
        elif intent.intent_type == IntentType.EXPLORATORY:
            base_count = 5
        else:
            base_count = 3

        # Expand if requested
        if expand:
            base_count = min(base_count + 2, self.num_topics)

        # For initial plan, select first N topics
        # In production, this would use query embedding similarity
        topic_ids = set(range(min(base_count, self.num_topics)))

        return topic_ids

    def _determine_chunk_count(self, intent: Intent) -> int:
        """Determine max chunks based on intent"""
        from core.contracts.intent import IntentType

        if intent.intent_type == IntentType.FACTUAL:
            return Config.TOP_K
        elif intent.intent_type == IntentType.ANALYTICAL:
            return Config.TOP_K + 2
        elif intent.intent_type == IntentType.COMPARATIVE:
            return Config.TOP_K + 3
        elif intent.intent_type == IntentType.PROCEDURAL:
            return Config.TOP_K + 2
        elif intent.intent_type == IntentType.EXPLORATORY:
            return Config.TOP_K + 5
        else:
            return Config.TOP_K

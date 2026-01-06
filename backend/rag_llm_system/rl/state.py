from config.thresholds import Intent, RLState


class StateBuilder:
    """Build minimal RL state."""

    @staticmethod
    def build(
        topic_id: int,
        intent: Intent,
        cache_hit: bool,
        last_evidence_score: float,
        contradiction_flag: bool,
        token_budget: int,
    ) -> RLState:
        """Create state representation."""
        return RLState(
            topic_id=topic_id,
            intent=intent,
            cache_hit=cache_hit,
            last_evidence_score=last_evidence_score,
            contradiction_flag=contradiction_flag,
            token_budget=token_budget,
        )

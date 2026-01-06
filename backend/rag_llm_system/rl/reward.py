class RewardComputer:
    """Compute RL rewards."""

    WEIGHTS = {
        "evidence": 1.0,
        "consistency": 0.5,
        "cache_hit": 0.3,
        "hallucination": -2.0,
        "contradiction": -2.5,
        "unnecessary_retrieval": -0.3,
        "latency": -0.2,
        "token_overuse": -0.2,
    }

    @staticmethod
    def compute(
        evidence_score: float,
        is_consistent: bool,
        cache_hit: bool,
        has_hallucination: bool,
        has_contradiction: bool,
        did_retrieval: bool,
        latency_ms: float,
        tokens_used: int,
        token_budget: int,
    ) -> float:
        """Compute total reward."""
        reward = 0.0

        # Positive rewards
        reward += RewardComputer.WEIGHTS["evidence"] * evidence_score
        reward += RewardComputer.WEIGHTS["consistency"] * (
            1.0 if is_consistent else 0.0
        )
        reward += RewardComputer.WEIGHTS["cache_hit"] * (1.0 if cache_hit else 0.0)

        # Negative penalties
        if has_hallucination:
            reward += RewardComputer.WEIGHTS["hallucination"]

        if has_contradiction:
            reward += RewardComputer.WEIGHTS["contradiction"]

        if did_retrieval and cache_hit:
            reward += RewardComputer.WEIGHTS["unnecessary_retrieval"]

        if latency_ms > 1000:
            reward += RewardComputer.WEIGHTS["latency"] * (latency_ms / 1000)

        if tokens_used > token_budget:
            reward += RewardComputer.WEIGHTS["token_overuse"]

        return reward

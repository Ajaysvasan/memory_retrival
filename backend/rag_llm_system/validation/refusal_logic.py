from config.thresholds import THRESHOLDS


class RefusalLogic:
    """Decide whether to refuse generation."""

    @staticmethod
    def should_refuse(
        evidence_score: float, has_contradiction: bool, retrieval_failed: bool
    ) -> bool:
        """Determine if we should refuse."""
        if has_contradiction:
            return True

        if retrieval_failed:
            return True

        if evidence_score < THRESHOLDS.REFUSAL_THRESHOLD:
            return True

        return False

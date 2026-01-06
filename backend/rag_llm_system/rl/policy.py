from config.thresholds import Action, Intent, RLState


class RLPolicy:
    """RL-based decision policy."""

    def __init__(self):
        self.rule_based = True  # Start with rules, then learn

    def decide_action(self, state: RLState) -> Action:
        """Decide what action to take."""
        if self.rule_based:
            return self._rule_based_policy(state)
        else:
            return self._learned_policy(state)

    def _rule_based_policy(self, state: RLState) -> Action:
        """Bootstrap rule-based policy."""
        # Refuse if contradiction detected
        if state.contradiction_flag:
            return Action.REFUSE

        # Skip retrieval if cache hit with high confidence
        if state.cache_hit and state.last_evidence_score > 0.8:
            return Action.SKIP_RETRIEVAL

        # Use episodic memory for follow-ups
        if state.intent == Intent.FOLLOW_UP:
            return Action.RETRIEVE_EM

        # Default: retrieve from LTM
        return Action.RETRIEVE_LTM

    def _learned_policy(self, state: RLState) -> Action:
        """Placeholder for learned policy."""
        # TODO: Implement learned policy network
        return self._rule_based_policy(state)

from typing import List

from memory.belief_store import BeliefStore


class ContradictionDetector:
    """Detect contradictions with beliefs."""

    def __init__(self, belief_store: BeliefStore):
        self.belief_store = belief_store

    def detect(self, claims: List[str]) -> bool:
        """Check for contradictions."""
        for claim in claims:
            if self.belief_store.check_contradiction(claim):
                return True
        return False

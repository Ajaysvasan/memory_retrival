import hashlib
from datetime import datetime
from typing import Dict, Optional

from config.thresholds import THRESHOLDS, Belief, BeliefStatus


class BeliefStore:
    """Long-term factual memory."""

    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}

    def add_belief(self, claim: str, evidence_refs: List[str], confidence: float):
        """Store high-confidence belief."""
        if confidence < THRESHOLDS.MIN_BELIEF_CONFIDENCE:
            return

        belief_id = hashlib.sha256(claim.encode()).hexdigest()[:16]
        self.beliefs[belief_id] = Belief(
            claim=claim,
            evidence_refs=evidence_refs,
            confidence=confidence,
            status=BeliefStatus.ACTIVE,
            timestamp=datetime.now(),
        )

    def check_contradiction(self, new_claim: str) -> Optional[Belief]:
        """Check for contradictions."""
        # Placeholder: implement semantic contradiction detection
        for belief in self.beliefs.values():
            if belief.status == BeliefStatus.ACTIVE:
                # Simple check: exact negation
                if self._is_negation(belief.claim, new_claim):
                    return belief
        return None

    def _is_negation(self, claim1: str, claim2: str) -> bool:
        """Check if claims contradict."""
        # Placeholder: implement NLI-based check
        return False

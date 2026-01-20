from dataclasses import dataclass
from typing import Optional
from core.contracts.intent import Intent
from core.contracts.retrieval_plan import RetrievalPlan


@dataclass(frozen=True)
class PhaseADecision:
    """Phase A orchestration decision"""

    intent: Intent
    plan: RetrievalPlan
    should_proceed: bool
    refusal_reason: Optional[str] = None

    def __post_init__(self):
        if not self.should_proceed and not self.refusal_reason:
            raise ValueError("Refusal reason required when should_proceed is False")

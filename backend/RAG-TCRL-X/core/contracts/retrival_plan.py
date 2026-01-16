import enum
from dataclasses import dataclass


class FailureAction(enum.Enum):
    REFUSE = "REFUSE"
    ASK_FOR_CLARIFICATION = "ASK_FOR_CLARIFICATION"


@dataclass(frozen=True)
class RetrievalPlan:
    initial_k: int = 15
    max_k: int = 100
    retrieval_allowed: bool = True
    on_failure: FailureAction = FailureAction.ASK_FOR_CLARIFICATION
    min_relevance_score: float = 0.5
    allow_expansion: bool = True
    allowed_topics: tuple[str, ...] = (
        "general",
        "technical",
        "financial",
        "health",
        "education",
    )

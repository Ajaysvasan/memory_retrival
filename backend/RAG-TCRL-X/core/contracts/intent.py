import enum
from dataclasses import dataclass


class IntentType(enum.Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"
    PROCEDURAL = "PROCEDURAL"
    EXPLANATORY = "EXPLANATORY"
    FACTUAL = "FACTUAL"


@dataclass(frozen=True)
class IntentFrame:
    intent: IntentType
    confidence_score: float
    is_actionable: bool

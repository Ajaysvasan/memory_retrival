from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IntentType(Enum):
    """Query intent categories"""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Intent:
    """Classified intent with confidence"""

    intent_type: IntentType
    confidence: float
    reasoning: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_confident(self) -> bool:
        """Check if intent classification is confident"""
        return self.confidence >= 0.7

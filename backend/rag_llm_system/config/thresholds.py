from dataclasses import dataclass

from settings import SystemConfig


# config/thresholds.py
@dataclass
class Thresholds:
    MIN_EVIDENCE_SCORE: float = 0.7
    CONTRADICTION_THRESHOLD: float = 0.8
    REFUSAL_THRESHOLD: float = 0.5
    MIN_BELIEF_CONFIDENCE: float = 0.8
    NLI_ENTAILMENT_THRESHOLD: float = 0.75


CONFIG = SystemConfig()
THRESHOLDS = Thresholds()


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Intent(Enum):
    FACTUAL = "factual"
    OPINION = "opinion"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"


class BeliefStatus(Enum):
    ACTIVE = "active"
    REVISED = "revised"
    DEPRECATED = "deprecated"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Belief:
    claim: str
    evidence_refs: List[str]
    confidence: float
    status: BeliefStatus
    timestamp: datetime
    variants_allowed: bool = False


@dataclass
class Query:
    text: str
    topic_id: Optional[int] = None
    intent: Optional[Intent] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RLState:
    topic_id: int
    intent: Intent
    cache_hit: bool
    last_evidence_score: float
    contradiction_flag: bool
    token_budget: int


class Action(Enum):
    SKIP_RETRIEVAL = "skip"
    RETRIEVE_LTM = "retrieve_ltm"
    RETRIEVE_EM = "retrieve_em"
    REFUSE = "refuse"


if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(sys.path)

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """RL-controlled actions"""

    USE_CACHE = "use_cache"
    RETRIEVE_ANN = "retrieve_ann"
    EXPAND_TOPIC_SET = "expand_topic_set"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Decision:
    """RL agent decision"""

    action: ActionType
    confidence: float
    state_features: tuple[float, ...]

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

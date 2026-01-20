from dataclasses import dataclass, field
from typing import List, Set, FrozenSet
from datetime import datetime


@dataclass(frozen=True)
class RetrievalPlan:
    """Immutable retrieval execution plan"""

    query_hash: str
    topic_ids: FrozenSet[int]
    max_chunks: int
    use_cache: bool
    use_ann: bool
    expand_topics: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")

        if not self.topic_ids:
            raise ValueError("topic_ids cannot be empty")

    @property
    def cache_key(self) -> str:
        """Generate cache key from plan"""
        topics_str = ",".join(map(str, sorted(self.topic_ids)))
        return f"{self.query_hash}::{topics_str}::{self.max_chunks}"

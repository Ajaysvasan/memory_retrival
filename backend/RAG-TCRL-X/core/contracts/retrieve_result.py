from dataclasses import dataclass
from typing import List, Optional
from core.contracts.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True)
class RetrieveResult:
    """Result of retrieval operation"""

    chunks: tuple[RetrievedChunk, ...]
    from_cache: bool
    total_searched: int
    retrieval_time_ms: float

    def __post_init__(self):
        if self.total_searched < 0:
            raise ValueError("total_searched must be non-negative")

        if self.retrieval_time_ms < 0:
            raise ValueError("retrieval_time_ms must be non-negative")

    @property
    def num_chunks(self) -> int:
        """Number of retrieved chunks"""
        return len(self.chunks)

    @property
    def avg_similarity(self) -> float:
        """Average similarity score"""
        if not self.chunks:
            return 0.0
        return sum(c.similarity_score for c in self.chunks) / len(self.chunks)

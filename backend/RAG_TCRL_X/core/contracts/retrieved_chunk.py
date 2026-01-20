from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk with metadata"""

    chunk_id: int
    text: str
    topic_id: int
    similarity_score: float
    chunk_index: int
    source_doc: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"Similarity score must be in [0, 1], got {self.similarity_score}"
            )

        if self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")

        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

    @property
    def is_relevant(self) -> bool:
        """Check if chunk meets relevance threshold"""
        from config import Config

        return self.similarity_score >= Config.SIMILARITY_THRESHOLD

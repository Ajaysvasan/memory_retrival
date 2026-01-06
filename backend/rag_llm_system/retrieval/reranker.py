from typing import List

from config.thresholds import Chunk, Query


class Reranker:
    """Lightweight reranking with heuristics."""

    @staticmethod
    def rerank(
        chunks: List[Chunk], query: Query, max_candidates: int = 20
    ) -> List[Chunk]:
        """Rerank retrieved chunks."""
        if len(chunks) <= max_candidates:
            return chunks

        # Simple heuristic: prefer longer chunks (more context)
        scored = [(chunk, len(chunk.text.split())) for chunk in chunks[:max_candidates]]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored]

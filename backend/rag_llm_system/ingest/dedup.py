import hashlib
from typing import List, Set

from config.thresholds import Chunk


class Deduplicator:
    """Hash-based exact deduplication."""

    @staticmethod
    def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
        """Remove exact duplicates in O(n)."""
        seen: Set[str] = set()
        unique_chunks = []

        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

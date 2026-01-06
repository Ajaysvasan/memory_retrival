import hashlib

from retrieval.intend_detector import Intent


class CacheKey:
    """Global cache key."""

    @staticmethod
    def create(topic_id: int, intent: Intent, query_text: str) -> str:
        """Create deterministic cache key."""
        query_sig = hashlib.sha256(query_text.encode()).hexdigest()[:8]
        return f"{topic_id}:{intent.value}:{query_sig}"

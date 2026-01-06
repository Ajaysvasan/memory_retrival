import time
from typing import Any, Dict, List, Optional

from config.thresholds import CONFIG


class L2Cache:
    """CPU-tier warm cache."""

    def __init__(self, max_size: int = CONFIG.L2_CACHE_SIZE):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.ttl = CONFIG.CACHE_TTL_SECONDS

    def get(self, key: str) -> Optional[List[str]]:
        """Get with TTL check."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["chunk_ids"]
            else:
                del self.cache[key]
        return None

    def put(self, key: str, chunk_ids: List[str]):
        """Put with size limit."""
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

        self.cache[key] = {"chunk_ids": chunk_ids, "timestamp": time.time()}

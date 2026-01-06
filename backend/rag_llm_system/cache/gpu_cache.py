import time
from collections import OrderedDict
from typing import List, Optional

from config.thresholds import CONFIG


class L1Cache:
    """GPU-tier hot cache."""

    def __init__(self, max_size: int = CONFIG.L1_CACHE_SIZE):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[str]]:
        """Get with LRU."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]["chunk_ids"]
        self.misses += 1
        return None

    def put(self, key: str, chunk_ids: List[str], score: float):
        """Put with admission control."""
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = {
            "chunk_ids": chunk_ids,
            "score": score,
            "timestamp": time.time(),
        }

    def evict_low_quality(self, min_score: float):
        """Correctness-based eviction."""
        to_remove = [k for k, v in self.cache.items() if v["score"] < min_score]
        for k in to_remove:
            del self.cache[k]

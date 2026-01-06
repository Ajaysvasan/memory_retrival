from typing import List, Optional

from cpu_cache import L2Cache
from gpu_cache import L1Cache


class CacheManager:
    """Tiered cache coordinator."""

    def __init__(self):
        self.l1 = L1Cache()
        self.l2 = L2Cache()

    def lookup(self, key: str) -> Optional[List[str]]:
        """Multi-tier lookup."""
        # Try L1
        result = self.l1.get(key)
        if result:
            return result

        # Try L2
        result = self.l2.get(key)
        if result:
            # Promote to L1
            self.l1.put(key, result, score=0.8)
            return result

        return None

    def store(self, key: str, chunk_ids: List[str], score: float):
        """Store in appropriate tier."""
        if score >= 0.8:
            self.l1.put(key, chunk_ids, score)
        self.l2.put(key, chunk_ids)

from collections import deque
from datetime import datetime
from typing import Dict, List

from config.thresholds import CONFIG


class EpisodicMemory:
    """Compressed past episodes."""

    def __init__(self, max_episodes: int = CONFIG.EM_MAX_EPISODES):
        self.episodes: deque = deque(maxlen=max_episodes)

    def add_episode(self, summary: str, metadata: Dict):
        """Store episode summary."""
        self.episodes.append(
            {"summary": summary, "metadata": metadata, "timestamp": datetime.now()}
        )

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Simple keyword search."""
        # Placeholder: implement proper semantic search
        return list(self.episodes)[:k]

from collections import deque
from datetime import datetime
from typing import Dict, List


class ShortTermMemory:
    """Recent conversation context."""

    def __init__(self, max_turns: int = CONFIG.STM_MAX_TURNS):
        self.turns: deque = deque(maxlen=max_turns)

    def add_turn(self, query: str, response: str):
        """Add conversation turn."""
        self.turns.append(
            {"query": query, "response": response, "timestamp": datetime.now()}
        )

    def get_context(self) -> List[Dict]:
        """Get recent context."""
        return list(self.turns)

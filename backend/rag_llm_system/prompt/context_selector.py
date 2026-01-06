from typing import Any, Dict, List

from config.thresholds import Chunk
from memory.belief_store import Belief
from memory.stm import ShortTermMemory
from memory.wm import WorkingMemory


class ContextSelector:
    """Select and prioritize context."""

    def __init__(self, token_budget: int = 2048):
        self.token_budget = token_budget

    def select_context(
        self,
        stm: ShortTermMemory,
        wm: WorkingMemory,
        chunks: List[Chunk],
        beliefs: List[Belief],
    ) -> Dict[str, Any]:
        """Select context within budget."""
        context = {
            "recent_turns": [],
            "constraints": [],
            "retrieved": [],
            "beliefs": [],
        }

        tokens_used = 0

        # Priority 1: Recent turns
        for turn in stm.get_context():
            turn_tokens = self._estimate_tokens(str(turn))
            if tokens_used + turn_tokens <= self.token_budget:
                context["recent_turns"].append(turn)
                tokens_used += turn_tokens

        # Priority 2: Constraints
        for constraint in wm.get_constraints():
            constraint_tokens = self._estimate_tokens(constraint)
            if tokens_used + constraint_tokens <= self.token_budget:
                context["constraints"].append(constraint)
                tokens_used += constraint_tokens

        # Priority 3: Retrieved chunks
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.text)
            if tokens_used + chunk_tokens <= self.token_budget:
                context["retrieved"].append(chunk.text)
                tokens_used += chunk_tokens

        return context

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation."""
        return len(text.split()) * 1.3

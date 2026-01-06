from typing import List

from config.thresholds import CONFIG


class WorkingMemory:
    """Current task state."""

    def __init__(self, max_tokens: int = CONFIG.WM_MAX_SIZE):
        self.constraints: List[str] = []
        self.max_tokens = max_tokens

    def add_constraint(self, constraint: str):
        """Add task constraint."""
        self.constraints.append(constraint)

    def get_constraints(self) -> List[str]:
        """Get current constraints."""
        return self.constraints

    def clear(self):
        """Reset working memory."""
        self.constraints = []

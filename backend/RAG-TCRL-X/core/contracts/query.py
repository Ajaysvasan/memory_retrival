from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Query:
    """Immutable query representation"""

    text: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")

        if len(self.text) > 10000:
            raise ValueError("Query text exceeds maximum length of 10000 characters")

    @property
    def normalized_text(self) -> str:
        """Return normalized query text"""
        return self.text.strip().lower()

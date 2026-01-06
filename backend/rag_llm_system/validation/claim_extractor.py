import re
from typing import List

from config.thresholds import Intent


class ClaimExtractor:
    """Extract factual claims from text."""

    @staticmethod
    def extract(text: str, intent: Intent) -> List[str]:
        """Extract claims based on intent."""
        if intent == Intent.OPINION:
            return []  # Skip validation for opinions

        # Simple sentence-based extraction
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]

        return claims

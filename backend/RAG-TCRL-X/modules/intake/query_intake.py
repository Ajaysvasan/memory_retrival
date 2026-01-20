import hashlib
from datetime import datetime
from typing import Optional

from core.contracts.query import Query
from logger import Logger


class QueryIntake:
    """Query ingestion and preprocessing"""

    def __init__(self):
        self.logger = Logger().get_logger("QueryIntake")

    def process(
        self, text: str, user_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> Query:
        """Process raw query text into Query object"""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        # Basic preprocessing
        text = text.strip()

        # Validate length
        if len(text) > 10000:
            raise ValueError("Query exceeds maximum length")

        query = Query(
            text=text, timestamp=datetime.now(), user_id=user_id, session_id=session_id
        )

        self.logger.debug(f"Processed query: {text[:100]}...")
        return query

    @staticmethod
    def compute_query_hash(query: Query) -> str:
        """Compute deterministic hash for query"""
        normalized = query.normalized_text
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

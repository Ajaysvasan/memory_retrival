from abc import ABC, abstractmethod
from core.contracts.query import Query
from core.contracts.intent import Intent


class IntentClassifier(ABC):
    """Abstract intent classifier interface"""

    @abstractmethod
    def classify(self, query: Query) -> Intent:
        """Classify query intent"""
        pass

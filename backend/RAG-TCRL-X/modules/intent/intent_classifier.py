from core.contracts.intent import IntentFrame
from core.contracts.query import QueryEnvelope


class IntentClassifier:
    def __init__(self):
        pass

    def classify(self, query: QueryEnvelope) -> IntentFrame:
        raise NotImplementedError("This method should be overridden by subclasses.")

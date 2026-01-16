from core.contracts.intent import IntentFrame, IntentType
from core.contracts.query import QueryEnvelope
from modules.intent.intent_classifier import IntentClassifier


class HeuristicIntentClassifier(IntentClassifier):
    MIN_QUERY_LENGTH = 3
    FACTUAL_KEYWORDS = ("what is", "who is", "define", "when did")
    PROCEDURAL_KEYWORDS = ("how to", "steps", "guide", "install", "configure", "build")
    EXPLANATORY_KEYWORDS = ("why", "explain", "how does", "what happens")

    def __init__(self):
        pass

    def classify(self, query: QueryEnvelope) -> IntentFrame:
        text = query.normalized_query.strip().lower()
        if not text:
            return IntentFrame(
                intent=IntentType.OUT_OF_SCOPE,
                confidence_score=0.0,
                is_actionable=False,
            )
        if any(k in text for k in self.PROCEDURAL_KEYWORDS):
            intent = IntentType.PROCEDURAL
            confidence_score = 0.8
        elif any(k in text for k in self.FACTUAL_KEYWORDS):
            intent = IntentType.FACTUAL
            confidence_score = 0.7

        elif any(k in text for k in self.EXPLANATORY_KEYWORDS):
            intent = IntentType.EXPLANATORY
            confidence_score = 0.7
        else:
            intent = IntentType.AMBIGUOUS
            confidence_score = 0.4
        word_count = len(text.split())
        if word_count < self.MIN_QUERY_LENGTH and intent != IntentType.OUT_OF_SCOPE:
            intent = IntentType.AMBIGUOUS
            confidence_score = min(confidence_score, 0.4)

        is_actionable = intent not in (IntentType.OUT_OF_SCOPE, IntentType.AMBIGUOUS)
        return IntentFrame(
            intent=intent,
            confidence_score=confidence_score,
            is_actionable=is_actionable,
        )

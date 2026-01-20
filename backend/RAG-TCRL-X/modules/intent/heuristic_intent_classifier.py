import re
from core.contracts.query import Query
from core.contracts.intent import Intent, IntentType
from modules.intent.intent_classifier import IntentClassifier
from logger import Logger


class HeuristicIntentClassifier(IntentClassifier):
    """Rule-based intent classification"""

    def __init__(self):
        self.logger = Logger().get_logger("IntentClassifier")

        # Intent patterns
        self.patterns = {
            IntentType.FACTUAL: [
                r"\b(what|who|when|where|which)\b",
                r"\b(is|are|was|were)\b.*\?",
                r"\b(define|definition)\b",
            ],
            IntentType.ANALYTICAL: [
                r"\b(why|how come)\b",
                r"\b(explain|analyze|interpret)\b",
                r"\b(reason|cause|because)\b",
            ],
            IntentType.COMPARATIVE: [
                r"\b(compare|contrast|difference|versus|vs)\b",
                r"\b(better|worse|more|less)\b.*than",
                r"\b(similar|different)\b",
            ],
            IntentType.PROCEDURAL: [
                r"\b(how to|how do|how can)\b",
                r"\b(steps|process|procedure|method)\b",
                r"\b(guide|tutorial|instructions)\b",
            ],
            IntentType.EXPLORATORY: [
                r"\b(explore|discover|find out)\b",
                r"\b(tell me about|information about)\b",
                r"\b(overview|summary)\b",
            ],
        }

    def classify(self, query: Query) -> Intent:
        """Classify query using pattern matching"""
        text = query.normalized_text

        # Score each intent type
        scores = {}
        for intent_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            scores[intent_type] = score

        # Get best match
        if max(scores.values()) == 0:
            intent_type = IntentType.UNKNOWN
            confidence = 0.3
        else:
            intent_type = max(scores, key=scores.get)
            # Confidence based on match strength
            total_patterns = sum(len(p) for p in self.patterns.values())
            confidence = min(0.95, 0.5 + (scores[intent_type] / total_patterns) * 0.5)

        intent = Intent(
            intent_type=intent_type,
            confidence=confidence,
            reasoning=f"Matched {scores.get(intent_type, 0)} patterns",
        )

        self.logger.debug(
            f"Classified intent: {intent_type.value} (confidence={confidence:.2f})"
        )
        return intent

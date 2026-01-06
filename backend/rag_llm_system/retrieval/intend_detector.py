from config.thresholds import Intent


class IntentDetector:
    """Rule-based intent classification."""

    QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}
    OPINION_WORDS = {"think", "believe", "feel", "opinion", "should"}
    CLARIFY_WORDS = {"mean", "explain", "clarify", "elaborate"}

    @staticmethod
    def detect(query_text: str) -> Intent:
        """Classify query intent."""
        text_lower = query_text.lower()
        words = set(text_lower.split())

        if any(w in words for w in IntentDetector.CLARIFY_WORDS):
            return Intent.CLARIFICATION

        if any(w in words for w in IntentDetector.OPINION_WORDS):
            return Intent.OPINION

        if any(text_lower.startswith(w) for w in IntentDetector.QUESTION_WORDS):
            return Intent.FACTUAL

        return Intent.FOLLOW_UP

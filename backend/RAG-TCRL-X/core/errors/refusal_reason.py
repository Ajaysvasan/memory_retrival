from enum import Enum


class RefusalReason(Enum):
    """Structured refusal reasons"""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    UNCERTAIN_INTENT = "uncertain_intent"
    NO_RELEVANT_DATA = "no_relevant_data"
    VALIDATION_FAILED = "validation_failed"
    SYSTEM_ERROR = "system_error"
    OVER_INFORMATION = "over_information"

    def to_message(self) -> str:
        """Convert to user-facing message"""
        messages = {
            RefusalReason.INSUFFICIENT_EVIDENCE: "I don't have enough evidence to answer this confidently.",
            RefusalReason.CONTRADICTION_DETECTED: "I found contradictory information and cannot provide a reliable answer.",
            RefusalReason.UNCERTAIN_INTENT: "I'm not sure what you're asking for. Could you rephrase?",
            RefusalReason.NO_RELEVANT_DATA: "I don't have relevant information to answer this question.",
            RefusalReason.VALIDATION_FAILED: "I cannot validate the answer against available evidence.",
            RefusalReason.SYSTEM_ERROR: "A system error occurred. Please try again.",
            RefusalReason.OVER_INFORMATION: "The answer would require more information than I can reliably provide.",
        }
        return messages.get(self, "I cannot answer this question.")

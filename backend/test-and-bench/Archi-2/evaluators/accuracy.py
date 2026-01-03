from difflib import SequenceMatcher
from collections import Counter
from core.metrics import AccuracyMetrics


class AccuracyEvaluator:
    """
    BLEU / ROUGE / F1 / Exact Match evaluation.
    """

    @staticmethod
    def evaluate(ai_answer: str, reference: str) -> AccuracyMetrics:
        ref = reference.lower().split()
        cand = ai_answer.lower().split()

        overlap = sum((Counter(ref) & Counter(cand)).values())
        precision = overlap / len(cand) if cand else 0
        recall = overlap / len(ref) if ref else 0

        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
        rouge = SequenceMatcher(None, reference, ai_answer).ratio()
        sim = SequenceMatcher(None, reference.lower(), ai_answer.lower()).ratio()

        return AccuracyMetrics(
            bleu_score=min(1.0, f1),
            rouge_score=min(1.0, rouge),
            similarity_score=min(1.0, sim),
            f1_score=min(1.0, f1),
            exact_match=reference.strip().lower() == ai_answer.strip().lower()
        )

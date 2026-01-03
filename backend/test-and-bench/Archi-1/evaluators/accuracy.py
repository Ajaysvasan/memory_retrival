from difflib import SequenceMatcher
from collections import Counter
from core.metrics import AccuracyMetrics


class AccuracyEvaluator:
    """
    Evaluates AI-generated answers against a reference answer.
    """

    @staticmethod
    def _bleu(reference: str, candidate: str) -> float:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        if not cand_tokens or not ref_tokens:
            return 0.0

        overlap = sum((Counter(ref_tokens) & Counter(cand_tokens)).values())
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        return min(1.0, 2 * precision * recall / (precision + recall))

    @staticmethod
    def _rouge(reference: str, candidate: str) -> float:
        return min(1.0, SequenceMatcher(None, reference, candidate).ratio())

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return min(1.0, SequenceMatcher(None, a.lower(), b.lower()).ratio())

    @staticmethod
    def _f1(reference: str, candidate: str) -> float:
        ref = set(reference.lower().split())
        cand = set(candidate.lower().split())

        if not ref or not cand:
            return 0.0

        inter = len(ref & cand)
        precision = inter / len(cand)
        recall = inter / len(ref)

        if precision + recall == 0:
            return 0.0

        return min(1.0, 2 * precision * recall / (precision + recall))

    def evaluate(self, ai_answer: str, reference_answer: str) -> AccuracyMetrics:
        return AccuracyMetrics(
            bleu_score=self._bleu(reference_answer, ai_answer),
            rouge_score=self._rouge(reference_answer, ai_answer),
            similarity_score=self._similarity(ai_answer, reference_answer),
            f1_score=self._f1(reference_answer, ai_answer),
            exact_match=ai_answer.strip().lower() == reference_answer.strip().lower(),
        )

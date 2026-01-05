from difflib import SequenceMatcher
from collections import Counter
from core.metrics import AccuracyMetrics


class AccuracyEvaluator:

    def evaluate(self, ai, ref):
        sim = SequenceMatcher(None, ai, ref).ratio()
        overlap = sum((Counter(ai.split()) & Counter(ref.split())).values())
        f1 = overlap / (len(ai.split()) + 1e-8)

        return AccuracyMetrics(sim, sim, sim, f1, ai.strip() == ref.strip())

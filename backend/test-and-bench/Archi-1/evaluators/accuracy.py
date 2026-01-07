from collections import Counter
from difflib import SequenceMatcher
from core.metrics import AccuracyMetrics

class AccuracyEvaluator:
    def evaluate(self, ai, human):
        overlap = sum((Counter(ai.split()) & Counter(human.split())).values())
        f1 = overlap / (len(ai.split())+1e-8)
        sim = SequenceMatcher(None, ai.lower(), human.lower()).ratio()
        return AccuracyMetrics(sim, sim, sim, f1, ai.strip()==human.strip())

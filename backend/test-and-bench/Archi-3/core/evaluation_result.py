from dataclasses import dataclass
from core.metrics import ConsistencyScore, ComparisonResult

@dataclass
class EvaluationResult:
    query: str
    consistency: ConsistencyScore
    comparison: ComparisonResult
    total_time: float

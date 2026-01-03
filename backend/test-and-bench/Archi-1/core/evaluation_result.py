from dataclasses import dataclass
from typing import List
from .retrieved_doc import RetrievedDoc
from .metrics import (
    ConsistencyScore,
    AccuracyMetrics,
    ComparisonResult,
    ComplexityMetrics,
)

@dataclass
class EvaluationResult:
    query: str
    retrieved_docs: List[RetrievedDoc]
    consistency: ConsistencyScore
    accuracy: AccuracyMetrics
    comparison: ComparisonResult
    complexity: List[ComplexityMetrics]
    total_time: float
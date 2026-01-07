from dataclasses import dataclass, field
from typing import List, Dict, Any
from core.retrieved_doc import RetrievedDoc
from core.metrics import (
    ConsistencyScore,
    AccuracyMetrics,
    ComparisonResult,
    ComplexityMetrics,
)

@dataclass
class EvaluationResult:
    query: str
    retrieved_docs: List[RetrievedDoc]
    generated_answer: str
    consistency: ConsistencyScore
    accuracy: AccuracyMetrics
    comparison: ComparisonResult
    complexity: List[ComplexityMetrics]
    total_time: float
    vector_db_stats: Dict[str, Any] = field(default_factory=dict)

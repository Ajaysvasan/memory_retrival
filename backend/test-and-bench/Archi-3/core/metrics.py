from dataclasses import dataclass

@dataclass
class ConsistencyScore:
    overall_consistency: float

@dataclass
class ComparisonResult:
    agreement_score: float
    recommendation: str

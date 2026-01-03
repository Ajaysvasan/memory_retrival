from dataclasses import dataclass

@dataclass
class ComplexityMetrics:
    operation: str
    time_complexity: str
    space_complexity: str
    description: str
    empirical_time: float = 0.0


@dataclass
class ConsistencyScore:
    semantic_consistency: float
    lexical_consistency: float
    overall_consistency: float
    variation_coefficient: float


@dataclass
class AccuracyMetrics:
    bleu_score: float
    rouge_score: float
    similarity_score: float
    f1_score: float
    exact_match: bool


@dataclass
class ComparisonResult:
    ai_answer: str
    human_answer: str
    semantic_similarity: float
    lexical_similarity: float
    structural_similarity: float
    length_ratio: float
    agreement_score: float
    recommendation: str

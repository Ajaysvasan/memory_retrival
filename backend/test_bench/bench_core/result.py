"""Result data structure for architecture outputs"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArchitectureResult:
    """Result from a single architecture"""
    architect_name: str
    input_query: str
    output: str
    latency: float  # in seconds
    accuracy: float
    confidence_score: float
    evidence_score: float
    average_accuracy: float
    error: Optional[str] = None

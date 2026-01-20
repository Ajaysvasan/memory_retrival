from dataclasses import dataclass
from typing import List, Optional, FrozenSet
from enum import Enum


class ValidationStatus(Enum):
    """Validation outcome"""

    VALID = "valid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    OVER_INFORMATION = "over_information"
    NO_CLAIMS = "no_claims"


@dataclass(frozen=True)
class Validation:
    """Validation result with evidence"""

    status: ValidationStatus
    evidence_score: float
    claims: tuple[str, ...]
    evidence_chunk_ids: FrozenSet[int]
    contradictions: tuple[str, ...] = ()
    reasoning: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.evidence_score <= 1.0:
            raise ValueError(
                f"Evidence score must be in [0, 1], got {self.evidence_score}"
            )

    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self.status == ValidationStatus.VALID

    @property
    def should_refuse(self) -> bool:
        """Check if system should refuse to answer"""
        return not self.is_valid

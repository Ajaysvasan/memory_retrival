from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    retrieval_score: float = 0.0
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

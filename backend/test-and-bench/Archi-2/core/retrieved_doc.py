from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    retrieval_score: float
    metadata: Dict[str, Any]

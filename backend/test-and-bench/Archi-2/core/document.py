from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

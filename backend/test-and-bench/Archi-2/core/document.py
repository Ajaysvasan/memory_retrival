from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
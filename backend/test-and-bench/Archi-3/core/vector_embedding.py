from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class VectorEmbedding:
    doc_id: str
    embedding: np.ndarray
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

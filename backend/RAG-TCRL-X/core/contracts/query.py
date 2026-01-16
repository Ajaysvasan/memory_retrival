from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class QueryEnvelope:
    raw_query: str
    normalized_query: str
    meta_data: Mapping[str, str]

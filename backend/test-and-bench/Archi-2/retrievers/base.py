from abc import ABC, abstractmethod
from typing import List, Tuple
from core.document import Document

class RetrieverModule(ABC):

    @abstractmethod
    def index(self, documents: List[Document]):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int):
        pass
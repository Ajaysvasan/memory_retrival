"""Base class for RAG architectures"""
from abc import ABC, abstractmethod
from typing import List
from core.document import Document
from core.result import ArchitectureResult


class BaseRAGArchitecture(ABC):
    """Base class for all RAG architectures"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents: List[Document] = []
        self.is_trained = False
    
    @abstractmethod
    def train(self, documents: List[Document]):
        """Train the architecture on documents"""
        pass
    
    @abstractmethod
    def query(self, query: str) -> ArchitectureResult:
        """Process a query and return result"""
        pass

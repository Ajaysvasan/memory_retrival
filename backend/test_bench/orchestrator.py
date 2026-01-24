"""Main orchestrator for test-bench"""
import logging
from pathlib import Path
from typing import List
from data_processor import DataProcessor
from architectures.architecture1_hybrid_rag import HybridRAGArchitecture
from architectures.architecture2_fid_rag import FiDRAGArchitecture
from architectures.architecture3_agentic_rag import AgenticRAGArchitecture
from core.document import Document
from core.result import ArchitectureResult


class TestBenchOrchestrator:
    """Orchestrates all three architectures"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize architectures
        self.arch1 = HybridRAGArchitecture()
        self.arch2 = FiDRAGArchitecture()
        self.arch3 = AgenticRAGArchitecture()
        
        self.documents: List[Document] = []
        self.is_trained = False
    
    def train_all(self):
        """Train all architectures on scraped data"""
        if self.is_trained:
            self.logger.info("Architectures already trained")
            return
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING ALL ARCHITECTURES")
        self.logger.info("=" * 80)
        
        # Load documents
        self.logger.info("Loading documents...")
        processor = DataProcessor(self.data_dir)
        self.documents = processor.load_documents()
        
        if not self.documents:
            raise ValueError("No documents loaded!")
        
        self.logger.info(f"Loaded {len(self.documents)} documents")
        
        # Train each architecture
        self.logger.info("Training Architecture 1...")
        self.arch1.train(self.documents)
        
        self.logger.info("Training Architecture 2...")
        self.arch2.train(self.documents)
        
        self.logger.info("Training Architecture 3...")
        self.arch3.train(self.documents)
        
        self.is_trained = True
        self.logger.info("=" * 80)
        self.logger.info("ALL ARCHITECTURES TRAINED")
        self.logger.info("=" * 80)
    
    def process_query(self, query: str) -> List[ArchitectureResult]:
        """Process query through all architectures"""
        if not self.is_trained:
            raise RuntimeError("Architectures not trained. Call train_all() first.")
        
        self.logger.info(f"Processing query: {query}")
        
        results = []
        
        # Process through each architecture
        self.logger.info("Running Architecture 1...")
        result1 = self.arch1.query(query)
        results.append(result1)
        
        self.logger.info("Running Architecture 2...")
        result2 = self.arch2.query(query)
        results.append(result2)
        
        self.logger.info("Running Architecture 3...")
        result3 = self.arch3.query(query)
        results.append(result3)
        
        return results

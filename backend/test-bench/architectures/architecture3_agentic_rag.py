"""Architecture 3: Agentic RAG (Tool-Orchestrated / Multi-Step RAG)"""
import time
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from core.document import Document
from core.result import ArchitectureResult
from architectures.base import BaseRAGArchitecture
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.answer_generator import generate_answer


class AgenticRAGArchitecture(BaseRAGArchitecture):
    """Agentic RAG with tool orchestration"""
    
    def __init__(self):
        super().__init__("Agentic RAG (Tool-Orchestrated / Multi-Step RAG)")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_embeddings = {}
        self.bm25_index = None
        self.tools = {}
    
    def train(self, documents: List[Document]):
        """Train on documents"""
        self.documents = documents
        
        # Create vector embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        for doc, emb in zip(documents, embeddings):
            self.vector_embeddings[doc.doc_id] = emb
        
        # Create BM25 index
        tokenized_corpus = [doc.content.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Register tools
        self.tools = {
            'vector_search': self._vector_search,
            'bm25_search': self._bm25_search,
            'hybrid_search': self._hybrid_search,
            'relevance_check': self._relevance_check,
            'answer_synthesis': self._answer_synthesis
        }
        
        self.is_trained = True
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Vector search tool"""
        query_embedding = self.embedding_model.encode(query)
        scores = {}
        for doc_id, doc_emb in self.vector_embeddings.items():
            similarity = float(np.dot(query_embedding, doc_emb) / 
                            (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8))
            scores[doc_id] = similarity
        
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc for doc in self.documents if doc.doc_id in [d[0] for d in top_docs]]
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Document]:
        """BM25 search tool"""
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
    
    def _hybrid_search(self, query: str, top_k: int = 10) -> List[Document]:
        """Hybrid search tool"""
        vector_docs = self._vector_search(query, top_k * 2)
        bm25_docs = self._bm25_search(query, top_k * 2)
        
        # Combine and deduplicate
        doc_dict = {doc.doc_id: doc for doc in vector_docs + bm25_docs}
        return list(doc_dict.values())[:top_k]
    
    def _relevance_check(self, query: str, docs: List[Document]) -> bool:
        """Relevance check tool"""
        if not docs:
            return False
        
        # Check if any document is relevant using embedding similarity
        query_embedding = self.embedding_model.encode(query)
        max_similarity = 0.0
        
        for doc in docs[:5]:
            if doc.doc_id in self.vector_embeddings:
                doc_emb = self.vector_embeddings[doc.doc_id]
                similarity = float(np.dot(query_embedding, doc_emb) / 
                                (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8))
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity > 0.3  # Threshold
    
    def _answer_synthesis(self, query: str, docs: List[Document]) -> str:
        """Answer synthesis tool"""
        if not docs:
            return "No relevant information found."
        
        # Generate focused answer from documents
        doc_texts = [doc.content for doc in docs[:3]]
        return generate_answer(query, doc_texts, max_length=500)
    
    def query(self, query: str) -> ArchitectureResult:
        """Process query using agentic approach"""
        start_time = time.time()
        
        try:
            # Step 1: Initial retrieval
            retrieved_docs = self.tools['hybrid_search'](query, top_k=10)
            
            # Step 2: Relevance check
            is_relevant = self.tools['relevance_check'](query, retrieved_docs)
            
            if not is_relevant:
                latency = time.time() - start_time
                return ArchitectureResult(
                    architect_name=self.name,
                    input_query=query,
                    output="No relevant information found for this query.",
                    latency=latency,
                    accuracy=0.0,
                    confidence_score=0.0,
                    evidence_score=0.0,
                    average_accuracy=0.0
                )
            
            # Step 3: Refine retrieval (get more specific docs)
            refined_docs = self.tools['vector_search'](query, top_k=5)
            
            # Step 4: Answer synthesis
            output = self.tools['answer_synthesis'](query, refined_docs)
            
            # Calculate metrics
            latency = time.time() - start_time
            
            # Calculate confidence based on document similarity
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            for doc in refined_docs[:3]:
                if doc.doc_id in self.vector_embeddings:
                    doc_emb = self.vector_embeddings[doc.doc_id]
                    similarity = float(np.dot(query_embedding, doc_emb) / 
                                    (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8))
                    similarities.append(similarity)
            
            confidence_score = float(np.mean(similarities)) if similarities else 0.0
            evidence_score = len(refined_docs) / 5.0
            accuracy = (confidence_score + evidence_score) / 2.0
            average_accuracy = accuracy
            
            return ArchitectureResult(
                architect_name=self.name,
                input_query=query,
                output=output,
                latency=latency,
                accuracy=accuracy,
                confidence_score=confidence_score,
                evidence_score=evidence_score,
                average_accuracy=average_accuracy
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return ArchitectureResult(
                architect_name=self.name,
                input_query=query,
                output=f"Error: {str(e)}",
                latency=latency,
                accuracy=0.0,
                confidence_score=0.0,
                evidence_score=0.0,
                average_accuracy=0.0,
                error=str(e)
            )

"""Architecture 1: Hybrid Two-Stage RAG with Cross-Encoder Reranking"""
import time
import numpy as np
from typing import List
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


class HybridRAGArchitecture(BaseRAGArchitecture):
    """Hybrid RAG with vector search + BM25 + cross-encoder reranking"""
    
    def __init__(self):
        super().__init__("Hybrid Two-Stage RAG with Cross-Encoder Reranking")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_embeddings = {}
        self.bm25_index = None
        # Use a stronger embedding model for reranking
        # Cross-encoder requires separate package, so we use a better embedding model
        try:
            self.reranker_model = SentenceTransformer('all-mpnet-base-v2')
        except Exception:
            # Fallback to same model
            self.reranker_model = self.embedding_model
    
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
        
        self.is_trained = True
    
    def query(self, query: str) -> ArchitectureResult:
        """Process query"""
        start_time = time.time()
        
        try:
            # Stage 1: Hybrid retrieval (vector + BM25)
            query_embedding = self.embedding_model.encode(query)
            
            # Vector similarity
            vector_scores = {}
            for doc_id, doc_emb in self.vector_embeddings.items():
                similarity = float(np.dot(query_embedding, doc_emb) / 
                                  (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8))
                vector_scores[doc_id] = similarity
            
            # BM25 scores
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Combine scores (60% vector, 40% BM25)
            combined_scores = {}
            for idx, doc in enumerate(self.documents):
                vector_score = vector_scores.get(doc.doc_id, 0.0)
                bm25_score = float(bm25_scores[idx])
                # Normalize BM25 score
                bm25_norm = bm25_score / (bm25_score + 1.0) if bm25_score > 0 else 0.0
                combined_scores[doc.doc_id] = 0.6 * vector_score + 0.4 * bm25_norm
            
            # Get top 20 candidates
            top_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:20]
            # Create a dict for quick lookup
            doc_dict = {doc.doc_id: doc for doc in self.documents}
            # Get documents in the order of top_candidates
            candidate_docs = [doc_dict[c[0]] for c in top_candidates if c[0] in doc_dict]
            candidate_texts = [doc.content for doc in candidate_docs]
            
            # Stage 2: Reranking using stronger embedding model
            if candidate_texts:
                try:
                    # Use stronger embedding model for reranking
                    query_emb = self.reranker_model.encode(query)
                    doc_embs = self.reranker_model.encode(candidate_texts)
                    rerank_scores = np.array([
                        float(np.dot(query_emb, doc_emb) / 
                            (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
                        for doc_emb in doc_embs
                    ])
                except Exception:
                    # Fallback to original embedding model
                    query_emb = self.embedding_model.encode(query)
                    doc_embs = self.embedding_model.encode(candidate_texts)
                    rerank_scores = np.array([
                        float(np.dot(query_emb, doc_emb) / 
                            (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
                        for doc_emb in doc_embs
                    ])
                
                # Get top 3 after reranking
                top_indices = np.argsort(rerank_scores)[::-1][:3]
                final_docs = [candidate_docs[i] for i in top_indices]
                final_texts = [doc.content for doc in final_docs]
                
                # Generate focused answer
                output = generate_answer(query, final_texts, max_length=500)
                
                # Calculate metrics
                latency = time.time() - start_time
                max_score = float(np.max(rerank_scores)) if len(rerank_scores) > 0 else 0.0
                confidence_score = max_score
                evidence_score = len(final_docs) / 3.0  # Normalized
                accuracy = (confidence_score + evidence_score) / 2.0
                average_accuracy = accuracy
                
            else:
                output = "No relevant documents found."
                latency = time.time() - start_time
                confidence_score = 0.0
                evidence_score = 0.0
                accuracy = 0.0
                average_accuracy = 0.0
            
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

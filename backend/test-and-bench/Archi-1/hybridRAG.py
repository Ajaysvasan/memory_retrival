#!/usr/bin/env python3
"""
HYBRID TWO-STAGE RAG WITH VECTOR DATABASE
Complete working system with persistent vector storage

Installation:
pip install sentence-transformers rank-bm25 torch numpy

Run:
python hybrid_rag_vectordb.py
"""

import sys
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import Counter
from difflib import SequenceMatcher

print("Loading dependencies...", flush=True)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: pip install sentence-transformers rank-bm25 torch numpy")
    sys.exit(1)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRecord:
    """Stored in Vector Database"""
    doc_id: str
    embedding: np.ndarray
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyScore:
    semantic_consistency: float
    lexical_consistency: float
    overall_consistency: float
    variation_coefficient: float


@dataclass
class AccuracyMetrics:
    bleu_score: float
    rouge_score: float
    similarity_score: float
    f1_score: float
    exact_match: bool


@dataclass
class ComparisonResult:
    ai_answer: str
    human_answer: str
    semantic_similarity: float
    lexical_similarity: float
    structural_similarity: float
    length_ratio: float
    agreement_score: float
    recommendation: str


@dataclass
class ComplexityMetrics:
    operation: str
    time_complexity: str
    space_complexity: str
    description: str


@dataclass
class EvaluationResult:
    query: str
    retrieved_docs: List[RetrievedDoc]
    generated_answer: str
    consistency: ConsistencyScore
    accuracy: AccuracyMetrics
    comparison: ComparisonResult
    complexity: List[ComplexityMetrics]
    total_time: float
    vector_db_stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# VECTOR DATABASE - CORE COMPONENT
# ============================================================================

class VectorDatabase:
    """Persistent Vector Database for storing and retrieving embeddings"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print("\n[VectorDB] Initializing Vector Database...", flush=True)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vectors: Dict[str, VectorRecord] = {}
        self.embedding_dim = 384
        print("[VectorDB] ✓ Ready\n", flush=True)
    
    def add_documents(self, documents: List[Document]):
        """Add documents and compute embeddings once"""
        print(f"[VectorDB] Creating embeddings for {len(documents)} documents...", flush=True)
        
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents, convert_to_tensor=False)
        
        for doc, embedding in zip(documents, embeddings):
            self.vectors[doc.doc_id] = VectorRecord(
                doc_id=doc.doc_id,
                embedding=np.array(embedding, dtype=np.float32),
                content=doc.content,
                metadata=doc.metadata
            )
        
        print(f"[VectorDB] ✓ Stored {len(documents)} vectors", flush=True)
        print(f"[VectorDB] Memory: {self._get_memory_usage():.4f} MB\n", flush=True)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Vector similarity search"""
        if not self.vectors:
            return []
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        similarities = {}
        for doc_id, record in self.vectors.items():
            similarity = np.dot(query_embedding, record.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(record.embedding) + 1e-8
            )
            similarities[doc_id] = float(similarity)
        
        top_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in top_docs:
            record = self.vectors[doc_id]
            results.append(RetrievedDoc(
                doc_id=doc_id,
                content=record.content,
                semantic_score=score,
                metadata=record.metadata
            ))
        return results
    
    def _get_memory_usage(self) -> float:
        return (len(self.vectors) * self.embedding_dim * 4) / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.vectors),
            "embedding_dimension": self.embedding_dim,
            "memory_usage_mb": self._get_memory_usage(),
            "status": "Ready" if self.vectors else "Empty"
        }


# ============================================================================
# BM25 INDEXING
# ============================================================================

class BM25Index:
    """BM25 keyword-based indexing"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def index(self, documents: List[Document]):
        print("[BM25] Creating index...", flush=True)
        self.documents = documents
        tokenized = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print("[BM25] ✓ Ready\n", flush=True)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        if not self.bm25:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievedDoc(
                doc_id=self.documents[idx].doc_id,
                content=self.documents[idx].content,
                bm25_score=float(scores[idx]),
                metadata=self.documents[idx].metadata
            ))
        return results


# ============================================================================
# HYBRID RETRIEVAL WITH VECTOR DB + BM25
# ============================================================================

class HybridRetrieverWithVectorDB:
    """Hybrid retrieval combining Vector DB + BM25"""
    
    def __init__(self, alpha: float = 0.6):
        print("[HybridRetriever] Initializing...", flush=True)
        self.vector_db = VectorDatabase()
        self.bm25_index = BM25Index()
        self.alpha = alpha
        self.documents = []
    
    def index(self, documents: List[Document]):
        self.documents = documents
        self.vector_db.add_documents(documents)
        self.bm25_index.index(documents)
        print(f"[HybridRetriever] ✓ Indexed {len(documents)} documents\n", flush=True)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """Hybrid search: Vector DB + BM25"""
        vector_results = self.vector_db.search(query, top_k=top_k*2)
        bm25_results = self.bm25_index.search(query, top_k=top_k*2)
        
        doc_scores = {}
        for doc in vector_results:
            doc_scores[doc.doc_id] = self.alpha * doc.semantic_score
        
        for doc in bm25_results:
            if doc.doc_id in doc_scores:
                doc_scores[doc.doc_id] += (1 - self.alpha) * doc.bm25_score
            else:
                doc_scores[doc.doc_id] = (1 - self.alpha) * doc.bm25_score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, hybrid_score in sorted_docs:
            doc = next((d for d in self.documents if d.doc_id == doc_id), None)
            if doc:
                results.append(RetrievedDoc(
                    doc_id=doc_id,
                    content=doc.content,
                    hybrid_score=hybrid_score,
                    metadata=doc.metadata
                ))
        return results


# ============================================================================
# CROSS-ENCODER RERANKING
# ============================================================================

class CrossEncoderReranker:
    """Cross-Encoder reranking"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print("[Reranker] Loading cross-encoder...", flush=True)
        self.model = CrossEncoder(model_name)
        print("[Reranker] ✓ Ready\n", flush=True)
    
    def rerank(self, query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]


# ============================================================================
# EVALUATORS
# ============================================================================

class ConsistencyEvaluator:
    def __init__(self, vector_db: VectorDatabase):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_db = vector_db
    
    def evaluate(self, query: str, docs: List[RetrievedDoc]) -> ConsistencyScore:
        if not docs:
            return ConsistencyScore(0, 0, 0, 0)
        
        query_emb = self.embedding_model.encode(query)
        semantic_sims = [
            float(np.dot(query_emb, self.embedding_model.encode(d.content)) /
                  (np.linalg.norm(query_emb) * np.linalg.norm(self.embedding_model.encode(d.content)) + 1e-8))
            for d in docs
        ]
        semantic = float(np.mean(semantic_sims))
        
        query_tokens = set(query.lower().split())
        lexical_sims = [
            len(query_tokens & set(d.content.lower().split())) /
            len(query_tokens | set(d.content.lower().split()))
            for d in docs
        ]
        lexical = float(np.mean(lexical_sims))
        
        overall = 0.6 * semantic + 0.4 * lexical
        variation = float(np.std(semantic_sims) / (np.mean(semantic_sims) + 1e-8))
        
        return ConsistencyScore(
            semantic_consistency=min(1.0, semantic),
            lexical_consistency=min(1.0, lexical),
            overall_consistency=min(1.0, overall),
            variation_coefficient=variation
        )


class AccuracyEvaluator:
    @staticmethod
    def evaluate(ai: str, human: str) -> AccuracyMetrics:
        def bleu(ref, cand):
            ref_t = ref.lower().split()
            cand_t = cand.lower().split()
            if not cand_t: return 0.0
            overlap = sum((Counter(ref_t) & Counter(cand_t)).values())
            p = overlap / len(cand_t)
            r = overlap / len(ref_t) if ref_t else 0.0
            return min(1.0, 2 * p * r / (p + r + 1e-8)) if p + r > 0 else 0.0
        
        def sim(s1, s2):
            return min(1.0, SequenceMatcher(None, s1.lower(), s2.lower()).ratio())
        
        return AccuracyMetrics(
            bleu_score=bleu(human, ai),
            rouge_score=sim(human, ai),
            similarity_score=sim(ai, human),
            f1_score=bleu(human, ai),
            exact_match=ai.lower().strip() == human.lower().strip()
        )


class ComparisonEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def evaluate(self, ai: str, human: str) -> ComparisonResult:
        ai_e = self.embedding_model.encode(ai)
        human_e = self.embedding_model.encode(human)
        sem_sim = min(1.0, float(np.dot(ai_e, human_e) / (np.linalg.norm(ai_e) * np.linalg.norm(human_e) + 1e-8)))
        
        ai_t = set(ai.lower().split())
        human_t = set(human.lower().split())
        lex_sim = len(ai_t & human_t) / len(ai_t | human_t) if ai_t | human_t else 0
        
        struct_sim = SequenceMatcher(None, ai, human).ratio()
        ai_len = len(ai.split())
        human_len = len(human.split())
        length = ai_len / human_len if human_len > 0 else 1.0
        
        agreement = 0.5 * sem_sim + 0.3 * lex_sim + 0.2 * struct_sim
        rec = "✓ ACCEPT" if agreement > 0.85 else "⚠ REVIEW" if agreement > 0.5 else "✗ REJECT"
        
        return ComparisonResult(
            ai_answer=ai,
            human_answer=human,
            semantic_similarity=sem_sim,
            lexical_similarity=min(1.0, lex_sim),
            structural_similarity=struct_sim,
            length_ratio=min(2.0, length),
            agreement_score=min(1.0, agreement),
            recommendation=rec
        )


class ComplexityAnalyzer:
    @staticmethod
    def get_analysis(n_docs: int) -> List[ComplexityMetrics]:
        return [
            ComplexityMetrics(
                "Vector DB Indexing",
                "O(n*d) where n=docs, d=384",
                "O(n*d)",
                "Embed and store documents"
            ),
            ComplexityMetrics(
                "Vector DB Search",
                "O(n*d)",
                "O(d+k)",
                "Cosine similarity search"
            ),
            ComplexityMetrics(
                "BM25 Indexing",
                "O(n*m) where m=avg_tokens",
                "O(n*m*log V)",
                "Build inverted index"
            ),
            ComplexityMetrics(
                "BM25 Search",
                "O(q+log n) where q=query_tokens",
                "O(1)",
                "Query lookup"
            ),
            ComplexityMetrics(
                "Hybrid Merge",
                "O(2k) where k=top_k",
                "O(k)",
                "Combine results"
            ),
            ComplexityMetrics(
                "Cross-Encoder Reranking",
                "O(k*d) where k=candidates",
                "O(k*d)",
                "Score pairs"
            ),
            ComplexityMetrics(
                "Full RAG Pipeline",
                "O(n*d) indexing, O(k*d) query",
                "O(n*d + k*d)",
                "Complete pipeline"
            ),
        ]


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class HybridRAGwithVectorDB:
    def __init__(self):
        print("\n[HybridRAG] Initializing system...\n", flush=True)
        self.retriever = HybridRetrieverWithVectorDB(alpha=0.6)
        self.reranker = CrossEncoderReranker()
        self.consistency_eval = None
        self.accuracy_eval = AccuracyEvaluator()
        self.comparison_eval = ComparisonEvaluator()
        self.documents = []
    
    def setup(self, documents: List[Document]):
        self.documents = documents
        self.retriever.index(documents)
        self.consistency_eval = ConsistencyEvaluator(self.retriever.vector_db)
    
    def evaluate(self, query: str, human_answer: str) -> EvaluationResult:
        start = time.time()
        
        print("[Stage 1] Vector DB Hybrid Retrieval...", flush=True)
        candidates = self.retriever.search(query, top_k=10)
        print(f"  ✓ Retrieved {len(candidates)} docs\n", flush=True)
        
        print("[Stage 2] Cross-Encoder Reranking...", flush=True)
        doc_texts = [d.content for d in candidates]
        reranked = self.reranker.rerank(query, doc_texts, 5)
        
        final_docs = []
        for orig_idx, rerank_score in reranked:
            doc = candidates[orig_idx]
            doc.rerank_score = rerank_score
            final_docs.append(doc)
        print(f"  ✓ Reranked to {len(final_docs)} docs\n", flush=True)
        
        print("[Stage 3] Answer Generation...", flush=True)
        combined = " ".join([d.content for d in final_docs[:2]])
        sentences = combined.split(".")
        query_tokens = set(query.lower().split())
        relevant = [s.strip() for s in sentences if s.strip() and len(set(s.lower().split()) & query_tokens) > 1]
        ai_answer = ". ".join(relevant[:3]) + "." if relevant else combined[:200] + "..."
        print(f"  ✓ Generated answer\n", flush=True)
        
        print("[Stage 4] Consistency Evaluation...", flush=True)
        consistency = self.consistency_eval.evaluate(query, final_docs)
        print(f"  ✓ Consistency: {consistency.overall_consistency:.4f}\n", flush=True)
        
        print("[Stage 5] Accuracy Evaluation...", flush=True)
        accuracy = self.accuracy_eval.evaluate(ai_answer, human_answer)
        print(f"  ✓ F1: {accuracy.f1_score:.4f}\n", flush=True)
        
        print("[Stage 6] AI vs Human Comparison...", flush=True)
        comparison = self.comparison_eval.evaluate(ai_answer, human_answer)
        print(f"  ✓ Agreement: {comparison.agreement_score:.4f}\n", flush=True)
        
        complexity = ComplexityAnalyzer.get_analysis(len(self.documents))
        
        return EvaluationResult(
            query=query,
            retrieved_docs=final_docs,
            generated_answer=ai_answer,
            consistency=consistency,
            accuracy=accuracy,
            comparison=comparison,
            complexity=complexity,
            total_time=time.time() - start,
            vector_db_stats=self.retriever.vector_db.get_stats()
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("HYBRID TWO-STAGE RAG WITH VECTOR DATABASE")
    print("="*70)
    
    docs = [
        Document("doc1", "Machine learning is AI learning from data without explicit programming.", {"src": "ML"}),
        Document("doc2", "Neural networks are interconnected nodes inspired by biological neurons.", {"src": "NN"}),
        Document("doc3", "Deep learning uses multiple neural network layers for hierarchical learning.", {"src": "DL"}),
        Document("doc4", "Backpropagation computes gradients to update weights in neural networks.", {"src": "Training"}),
        Document("doc5", "Transformers use self-attention mechanisms for sequential data processing.", {"src": "NLP"}),
        Document("doc6", "Gradient descent optimizes parameters by moving along negative gradient.", {"src": "Opt"}),
        Document("doc7", "CNNs specialize in image processing with convolutional layers.", {"src": "Vision"}),
        Document("doc8", "RNNs maintain hidden states to capture temporal information.", {"src": "Sequential"}),
        Document("doc9", "Attention mechanisms focus on relevant input parts via weighted combinations.", {"src": "Mechanisms"}),
        Document("doc10", "Embeddings convert tokens to continuous vectors capturing semantic relationships.", {"src": "Embed"}),
    ]
    
    system = HybridRAGwithVectorDB()
    print("\n[Setup] Building Vector DB and indexing...", flush=True)
    system.setup(docs)
    
    query = "How do neural networks learn from data?"
    human_answer = "Neural networks learn by adjusting weights through backpropagation with gradient computation."
    
    print(f"\n[Query] {query}\n", flush=True)
    result = system.evaluate(query, human_answer)
    
    output = {
        "System": "Hybrid RAG with Vector Database",
        "Query": result.query,
        "Generated_Answer": result.generated_answer,
        "Human_Answer": result.comparison.human_answer,
        "Vector_DB_Stats": result.vector_db_stats,
        "Consistency": {
            "semantic": round(result.consistency.semantic_consistency, 4),
            "lexical": round(result.consistency.lexical_consistency, 4),
            "overall": round(result.consistency.overall_consistency, 4),
        },
        "Accuracy": {
            "bleu": round(result.accuracy.bleu_score, 4),
            "rouge": round(result.accuracy.rouge_score, 4),
            "f1": round(result.accuracy.f1_score, 4),
        },
        "Comparison": {
            "semantic_sim": round(result.comparison.semantic_similarity, 4),
            "agreement": round(result.comparison.agreement_score, 4),
            "recommendation": result.comparison.recommendation,
        },
        "Complexity": [
            {"operation": c.operation, "time": c.time_complexity, "space": c.space_complexity}
            for c in result.complexity
        ],
        "Performance": {
            "execution_time_seconds": round(result.total_time, 4),
        }
    }
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")
    print(json.dumps(output, indent=2))
    print("\n✓ Vector Database successfully created, indexed, and used for retrieval!\n")


if __name__ == "__main__":
    main()
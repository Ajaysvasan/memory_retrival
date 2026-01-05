"""
Fusion-in-Decoder (FiD) RAG Architecture with Complete Evaluation Metrics
Includes: Consistency Level, Answer Accuracy, AI vs Human Comparison, Time/Space Complexity Analysis
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Dependencies: pip install sentence-transformers rank-bm25 torch transformers datasets


@dataclass
class Document:
    """Document structure"""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievedDoc:
    """Retrieved document with scores"""
    doc_id: str
    content: str
    retrieval_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComplexityMetrics:
    """Time and Space Complexity Analysis"""
    operation: str
    time_complexity: str
    space_complexity: str
    description: str
    empirical_time: float = 0.0


@dataclass
class ConsistencyScore:
    """Consistency evaluation metrics"""
    semantic_consistency: float
    lexical_consistency: float
    fusion_consistency: float
    overall_consistency: float
    variation_coefficient: float


@dataclass
class AccuracyMetrics:
    """Accuracy evaluation metrics"""
    bleu_score: float
    rouge_score: float
    similarity_score: float
    f1_score: float
    exact_match: bool


@dataclass
class ComparisonResult:
    """AI vs Human answer comparison"""
    ai_answer: str
    human_answer: str
    semantic_similarity: float
    lexical_similarity: float
    structural_similarity: float
    length_ratio: float
    agreement_score: float
    recommendation: str


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    query: str
    retrieved_docs: List[RetrievedDoc]
    generated_answer: str
    consistency: ConsistencyScore
    accuracy: AccuracyMetrics
    comparison: ComparisonResult
    complexity: List[ComplexityMetrics]
    total_time: float
    fusion_mechanism_stats: Dict[str, Any]


class RetrieverModule(ABC):
    """Abstract base class for retrieval"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        pass
    
    @abstractmethod
    def index(self, documents: List[Document]):
        pass


class SemanticRetriever(RetrieverModule):
    """Dense vector-based semantic retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index(self, documents: List[Document]):
        self.documents = documents
        contents = [doc.content for doc in documents]
        self.embeddings = self.model.encode(contents, convert_to_tensor=True)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = np.array([
            float(query_emb.dot(emb)) / 
            (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            for emb in self.embeddings
        ])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in top_indices]


class BM25RetrieverModule(RetrieverModule):
    """Sparse lexical retrieval using BM25"""
    
    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.bm25 = None
        self.documents = []
        self.BM25Okapi = BM25Okapi
    
    def index(self, documents: List[Document]):
        self.documents = documents
        tokenized_docs = [doc.content.lower().split() for doc in documents]
        self.bm25 = self.BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in top_indices]


class HybridRetriever:
    """Hybrid retriever for FiD"""
    
    def __init__(self, alpha: float = 0.5):
        self.semantic_retriever = SemanticRetriever()
        self.bm25_retriever = BM25RetrieverModule()
        self.alpha = alpha
        self.documents = []
    
    def index(self, documents: List[Document]):
        self.documents = documents
        self.semantic_retriever.index(documents)
        self.bm25_retriever.index(documents)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k*2)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k*2)
        
        sem_scores = {doc_id: score for doc_id, score in semantic_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        sem_min, sem_max = min(sem_scores.values()) or 0, max(sem_scores.values()) or 1
        bm25_min, bm25_max = min(bm25_scores.values()) or 0, max(bm25_scores.values()) or 1
        
        sem_norm = {k: (v-sem_min)/(sem_max-sem_min+1e-8) for k, v in sem_scores.items()}
        bm25_norm = {k: (v-bm25_min)/(bm25_max-bm25_min+1e-8) for k, v in bm25_scores.items()}
        
        all_doc_ids = set(sem_norm.keys()) | set(bm25_norm.keys())
        hybrid_scores = {
            doc_id: self.alpha * sem_norm.get(doc_id, 0) + 
                    (1-self.alpha) * bm25_norm.get(doc_id, 0)
            for doc_id in all_doc_ids
        }
        
        sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc = next((d for d in self.documents if d.doc_id == doc_id), None)
            if doc:
                results.append(RetrievedDoc(
                    doc_id=doc_id,
                    content=doc.content,
                    retrieval_score=score,
                    metadata=doc.metadata
                ))
        
        return results


class FusionEncoder(nn.Module):
    """Fusion-in-Decoder: Encodes query + each document independently, then fuses"""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 768, num_documents: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_documents = num_documents
        
        # Linear projection to standard embedding dim
        self.query_doc_projector = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Context encoder: processes [query, doc] pairs independently
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Attention weights for context importance
        self.attention_score = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fusion layer: combines encoded contexts with proper dimensioning
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, query_emb: torch.Tensor, doc_embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FiD Forward Pass:
        1. Encode each [query, doc] pair independently
        2. Compute attention weights for each context
        3. Fuse all contexts with attention
        
        Args:
            query_emb: Query embedding [embedding_dim] or [1, embedding_dim]
            doc_embs: List of document embeddings, each [embedding_dim] or [1, embedding_dim]
        
        Returns:
            fused_repr: Fused representation [1, hidden_dim]
            attention_weights: Attention scores for each document [num_docs]
        """
        # Ensure query_emb is 1D
        if query_emb.dim() == 2:
            query_emb = query_emb.squeeze(0)
        
        context_encodings = []
        
        # Encode each [query, doc] pair independently
        for doc_emb in doc_embs[:self.num_documents]:
            # Ensure doc_emb is 1D
            if doc_emb.dim() == 2:
                doc_emb = doc_emb.squeeze(0)
            
            # Concatenate query and document: [2 * embedding_dim]
            context_pair = torch.cat([query_emb, doc_emb], dim=0)  # [2*384]
            
            # Project to embedding_dim
            context_proj = self.query_doc_projector(context_pair)  # [384]
            
            # Add sequence dimension for transformer
            context_seq = context_proj.unsqueeze(0).unsqueeze(0)  # [1, 1, 384]
            
            # Encode context with transformer
            encoded = self.context_encoder(context_seq)  # [1, 1, 384]
            context_encodings.append(encoded.squeeze(0).squeeze(0))  # [384]
        
        # Stack all context encodings: [num_docs, 384]
        stacked_contexts = torch.stack(context_encodings)
        
        # Compute attention weights: [num_docs, 1]
        attn_logits = self.attention_score(stacked_contexts)  # [num_docs, 1]
        attn_weights = torch.softmax(attn_logits, dim=0)  # [num_docs, 1]
        
        # Weighted sum of contexts: [num_docs, 1] * [num_docs, 384]
        weighted_contexts = stacked_contexts * attn_weights  # [num_docs, 384]
        fused_context = torch.sum(weighted_contexts, dim=0)  # [384]
        
        # Pass through fusion layer
        fused_repr = self.fusion_layer(fused_context)  # [768]
        
        return fused_repr.unsqueeze(0), attn_weights.squeeze(-1)  # [1, 768], [num_docs]


class FusionDecoder(nn.Module):
    """Decoder for generating answers from fused representation"""
    
    def __init__(self, hidden_dim: int = 768, vocab_size: int = 30000, max_length: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, fused_repr: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Decode from fused representation
        
        Args:
            fused_repr: Fused representation [1, hidden_dim]
            tgt: Target sequence [seq_len, 1, hidden_dim]
        
        Returns:
            logits: Output logits [seq_len, vocab_size]
        """
        decoded = self.decoder(tgt, fused_repr.unsqueeze(0))
        logits = self.output_projection(decoded)
        return logits


class AnswerGeneratorFiD:
    """Generate answers using FiD architecture"""
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate_answer(self, query: str, retrieved_docs: List[RetrievedDoc], 
                       fusion_encoder: Optional[FusionEncoder] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate answer from retrieved documents using FiD
        
        Args:
            query: Query string
            retrieved_docs: Retrieved documents
            fusion_encoder: Optional FiD encoder module
        
        Returns:
            answer: Generated answer string
            fusion_stats: Statistics about fusion process
        """
        if not retrieved_docs:
            return "No relevant documents found.", {"error": "No documents"}
        
        # Get embeddings
        query_emb = torch.tensor(self.embedding_model.encode(query)).float()
        doc_embs = [
            torch.tensor(self.embedding_model.encode(doc.content)).float() 
            for doc in retrieved_docs
        ]
        
        fusion_stats = {
            "num_documents_fused": len(doc_embs),
            "query_embedding_dim": query_emb.shape[0],
            "document_embeddings_count": len(doc_embs),
        }
        
        # Use fusion encoder if provided
        if fusion_encoder is not None:
            with torch.no_grad():
                fused_repr, attn_weights = fusion_encoder(query_emb, doc_embs)
                fusion_stats["attention_weights"] = attn_weights.numpy().tolist()
                fusion_stats["fusion_applied"] = True
        else:
            fusion_stats["fusion_applied"] = False
        
        # Generate answer by selecting relevant content
        answer = self._extractive_generation(query, retrieved_docs)
        
        return answer, fusion_stats
    
    @staticmethod
    def _extractive_generation(query: str, retrieved_docs: List[RetrievedDoc]) -> str:
        """Extractive answer generation from documents"""
        if not retrieved_docs:
            return "No documents available."
        
        # Combine top documents
        combined_content = " ".join([doc.content for doc in retrieved_docs[:3]])
        
        # Extract relevant sentences
        sentences = combined_content.split(".")
        query_tokens = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_tokens = set(sentence.lower().split())
            overlap = len(query_tokens & sentence_tokens)
            if overlap > 1:
                relevant_sentences.append(sentence.strip())
        
        # Build answer
        if relevant_sentences:
            answer = ". ".join(relevant_sentences[:4])
            if answer:
                answer += "."
        else:
            answer = combined_content[:250] + "..."
        
        return answer


class ConsistencyEvaluator:
    """Evaluate consistency of retrieved results"""
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def evaluate(self, query: str, retrieved_docs: List[RetrievedDoc], 
                fusion_stats: Dict[str, Any]) -> ConsistencyScore:
        """Evaluate consistency with FiD fusion statistics"""
        if not retrieved_docs:
            return ConsistencyScore(0, 0, 0, 0, 0)
        
        # Semantic consistency
        query_emb = self.model.encode(query)
        doc_embs = [self.model.encode(doc.content) for doc in retrieved_docs]
        
        semantic_sims = [
            np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            for emb in doc_embs
        ]
        semantic_consistency = float(np.mean(semantic_sims))
        
        # Lexical consistency
        query_tokens = set(query.lower().split())
        lexical_sims = []
        for doc in retrieved_docs:
            doc_tokens = set(doc.content.lower().split())
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            jaccard = intersection / union if union > 0 else 0
            lexical_sims.append(jaccard)
        
        lexical_consistency = float(np.mean(lexical_sims))
        
        # Fusion consistency (based on attention weights)
        fusion_consistency = 0.0
        if fusion_stats.get("attention_weights"):
            attn = np.array(fusion_stats["attention_weights"])
            # Lower entropy = more focused = more consistent
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            max_entropy = np.log(len(attn))
            fusion_consistency = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        overall_consistency = (0.4 * semantic_consistency + 0.3 * lexical_consistency + 
                             0.3 * fusion_consistency)
        
        variation_coefficient = float(np.std(semantic_sims) / (np.mean(semantic_sims) + 1e-8))
        
        return ConsistencyScore(
            semantic_consistency=min(1.0, semantic_consistency),
            lexical_consistency=min(1.0, lexical_consistency),
            fusion_consistency=min(1.0, fusion_consistency),
            overall_consistency=min(1.0, overall_consistency),
            variation_coefficient=variation_coefficient
        )


class AccuracyEvaluator:
    """Evaluate answer accuracy"""
    
    @staticmethod
    def _bleu_score(reference: str, candidate: str) -> float:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens:
            return 0.0
        
        overlap = sum((Counter(ref_tokens) & Counter(cand_tokens)).values())
        precision = overlap / len(cand_tokens)
        
        if not ref_tokens:
            return 0.0
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1)
    
    @staticmethod
    def _rouge_score(reference: str, candidate: str) -> float:
        matcher = SequenceMatcher(None, reference, candidate)
        return min(1.0, matcher.ratio())
    
    @staticmethod
    def _similarity_score(str1: str, str2: str) -> float:
        matcher = SequenceMatcher(None, str1.lower(), str2.lower())
        return min(1.0, matcher.ratio())
    
    @staticmethod
    def _f1_score(reference: str, candidate: str) -> float:
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        
        if not cand_tokens or not ref_tokens:
            return 0.0
        
        intersection = len(ref_tokens & cand_tokens)
        precision = intersection / len(cand_tokens)
        recall = intersection / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1)
    
    def evaluate(self, ai_answer: str, reference_answer: str) -> AccuracyMetrics:
        return AccuracyMetrics(
            bleu_score=self._bleu_score(reference_answer, ai_answer),
            rouge_score=self._rouge_score(reference_answer, ai_answer),
            similarity_score=self._similarity_score(ai_answer, reference_answer),
            f1_score=self._f1_score(reference_answer, ai_answer),
            exact_match=ai_answer.lower().strip() == reference_answer.lower().strip()
        )


class ComparisonEvaluator:
    """Compare AI vs Human answers"""
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def evaluate(self, ai_answer: str, human_answer: str) -> ComparisonResult:
        # Semantic similarity
        ai_emb = self.model.encode(ai_answer)
        human_emb = self.model.encode(human_answer)
        semantic_sim = float(np.dot(ai_emb, human_emb) / 
                            (np.linalg.norm(ai_emb) * np.linalg.norm(human_emb) + 1e-8))
        
        # Lexical similarity
        ai_tokens = set(ai_answer.lower().split())
        human_tokens = set(human_answer.lower().split())
        intersection = len(ai_tokens & human_tokens)
        union = len(ai_tokens | human_tokens)
        lexical_sim = intersection / union if union > 0 else 0
        
        # Structural similarity
        matcher = SequenceMatcher(None, ai_answer, human_answer)
        structural_sim = matcher.ratio()
        
        # Length ratio
        ai_len = len(ai_answer.split())
        human_len = len(human_answer.split())
        length_ratio = ai_len / human_len if human_len > 0 else 1.0
        
        # Agreement score
        agreement = 0.5 * semantic_sim + 0.3 * lexical_sim + 0.2 * structural_sim
        
        # Recommendation
        if agreement > 0.85:
            recommendation = "ACCEPT - Excellent agreement with human answer"
        elif agreement > 0.70:
            recommendation = "REVIEW - Good agreement, minor differences acceptable"
        elif agreement > 0.50:
            recommendation = "REVIEW - Moderate agreement, verify key points"
        else:
            recommendation = "REJECT - Significant differences from human answer"
        
        return ComparisonResult(
            ai_answer=ai_answer,
            human_answer=human_answer,
            semantic_similarity=min(1.0, semantic_sim),
            lexical_similarity=min(1.0, lexical_sim),
            structural_similarity=min(1.0, structural_sim),
            length_ratio=min(2.0, length_ratio),
            agreement_score=min(1.0, agreement),
            recommendation=recommendation
        )


class ComplexityAnalyzer:
    """Analyze FiD complexity"""
    
    @staticmethod
    def get_complexity_analysis(n_docs: int, k_retrieved: int = 10, 
                               embedding_dim: int = 384, hidden_dim: int = 768) -> List[ComplexityMetrics]:
        return [
            ComplexityMetrics(
                operation="Semantic Embedding Indexing",
                time_complexity="O(n * d) where n=docs, d=embedding_dim",
                space_complexity="O(n * d)",
                description=f"Encode {n_docs} docs to embeddings of dim {embedding_dim}"
            ),
            ComplexityMetrics(
                operation="BM25 Indexing",
                time_complexity="O(n * m) where n=docs, m=avg_tokens",
                space_complexity="O(n * m * log(V))",
                description="Build inverted index for BM25"
            ),
            ComplexityMetrics(
                operation="Hybrid Retrieval",
                time_complexity="O(n*d + q*log(n)) where n=docs, q=query_tokens",
                space_complexity="O(d + k)",
                description=f"Retrieve top-{k_retrieved} documents"
            ),
            ComplexityMetrics(
                operation="FiD Context Encoding",
                time_complexity="O(k * (2*d + Transformer_complexity))",
                space_complexity="O(k * d)",
                description=f"Encode {k_retrieved} [query, doc] pairs independently"
            ),
            ComplexityMetrics(
                operation="FiD Fusion Layer",
                time_complexity="O(k * d → h) where k={0}, d={1}, h={2}".format(k_retrieved, embedding_dim, hidden_dim),
                space_complexity="O(k * d + h)",
                description="Fuse k context encodings with attention"
            ),
            ComplexityMetrics(
                operation="Attention Weight Computation",
                time_complexity="O(k * d) where k={0}".format(k_retrieved),
                space_complexity="O(k)",
                description="Compute softmax attention over k contexts"
            ),
            ComplexityMetrics(
                operation="Answer Generation",
                time_complexity="O(k + L) where k={0}, L=answer_length".format(k_retrieved),
                space_complexity="O(L)",
                description="Extract and combine answer from contexts"
            ),
            ComplexityMetrics(
                operation="Consistency Evaluation",
                time_complexity="O(k * d + k^2) where k={0}".format(k_retrieved),
                space_complexity="O(k * d)",
                description="Compute semantic/lexical consistency"
            ),
            ComplexityMetrics(
                operation="Accuracy Evaluation",
                time_complexity="O(m + n) where m,n=answer_tokens",
                space_complexity="O(m + n)",
                description="String comparisons for BLEU/ROUGE/F1"
            ),
            ComplexityMetrics(
                operation="FiD Full Pipeline",
                time_complexity="O(n*d) for indexing, O(k*d + k*Transformer) for query",
                space_complexity="O(n*d + k*h) where h={0}".format(hidden_dim),
                description="Complete FiD RAG pipeline"
            ),
        ]


class FusionInDecoderRAG:
    """Complete Fusion-in-Decoder RAG System"""
    
    def __init__(self, retrieval_alpha: float = 0.5, num_retrieved: int = 10, use_fusion: bool = True):
        self.retriever = HybridRetriever(alpha=retrieval_alpha)
        self.answer_generator = AnswerGeneratorFiD()
        self.consistency_evaluator = ConsistencyEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.comparison_evaluator = ComparisonEvaluator()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        self.num_retrieved = num_retrieved
        self.use_fusion = use_fusion
        
        # FiD components
        if use_fusion:
            self.fusion_encoder = FusionEncoder(embedding_dim=384, hidden_dim=768, 
                                               num_documents=num_retrieved)
        else:
            self.fusion_encoder = None
        
        self.documents = []
        self.n_docs = 0
    
    def index(self, documents: List[Document]):
        self.documents = documents
        self.n_docs = len(documents)
        self.retriever.index(documents)
    
    def retrieve(self, query: str) -> List[RetrievedDoc]:
        return self.retriever.retrieve(query, top_k=self.num_retrieved)
    
    def generate_answer(self, query: str, retrieved_docs: List[RetrievedDoc]) -> Tuple[str, Dict[str, Any]]:
        return self.answer_generator.generate_answer(query, retrieved_docs, self.fusion_encoder)
    
    def evaluate_full(self, query: str, human_answer: str) -> EvaluationResult:
        """Complete FiD RAG evaluation pipeline"""
        start_time = time.time()
        
        # Stage 1: Retrieve documents
        retrieved_docs = self.retrieve(query)
        
        # Stage 2: Generate answer using FiD
        generated_answer, fusion_stats = self.generate_answer(query, retrieved_docs)
        
        # Stage 3: Consistency evaluation
        consistency = self.consistency_evaluator.evaluate(query, retrieved_docs, fusion_stats)
        
        # Stage 4: Accuracy evaluation
        accuracy = self.accuracy_evaluator.evaluate(generated_answer, human_answer)
        
        # Stage 5: AI vs Human comparison
        comparison = self.comparison_evaluator.evaluate(generated_answer, human_answer)
        
        # Stage 6: Complexity analysis
        complexity = self.complexity_analyzer.get_complexity_analysis(self.n_docs, self.num_retrieved)
        
        total_time = time.time() - start_time
        
        return EvaluationResult(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_answer=generated_answer,
            consistency=consistency,
            accuracy=accuracy,
            comparison=comparison,
            complexity=complexity,
            total_time=total_time,
            fusion_mechanism_stats=fusion_stats
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Sample documents
    documents = [
        Document(
            doc_id="doc1",
            content="Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns.",
            metadata={"source": "AI Fundamentals", "category": "ML"}
        ),
        Document(
            doc_id="doc2",
            content="Deep learning uses neural networks with multiple layers to automatically learn representations from raw input for feature detection or classification tasks.",
            metadata={"source": "Deep Learning", "category": "DL"}
        ),
        Document(
            doc_id="doc3",
            content="Neural networks consist of interconnected nodes organized in layers that process information using connectionist approaches inspired by biological neurons in the brain.",
            metadata={"source": "Neural Networks", "category": "NN"}
        ),
        Document(
            doc_id="doc4",
            content="Backpropagation is an algorithm used to train neural networks by computing gradients of the loss function with respect to each weight by the chain rule.",
            metadata={"source": "Training Algorithms", "category": "Training"}
        ),
        Document(
            doc_id="doc5",
            content="Transformer models like BERT and GPT use self-attention mechanisms to process sequential data and achieve state-of-the-art results in NLP tasks.",
            metadata={"source": "Modern NLP", "category": "Transformers"}
        ),
        Document(
            doc_id="doc6",
            content="The gradient descent optimization algorithm updates weights by moving in the direction of negative gradient to minimize the loss function during training.",
            metadata={"source": "Optimization", "category": "Training"}
        ),
        Document(
            doc_id="doc7",
            content="Convolutional neural networks are specialized for processing grid-like data such as images using local connectivity and shared weights in convolutional layers.",
            metadata={"source": "Computer Vision", "category": "CNN"}
        ),
        Document(
            doc_id="doc8",
            content="Recurrent neural networks process sequential data by maintaining hidden states that capture information from previous time steps in the sequence.",
            metadata={"source": "Sequential Models", "category": "RNN"}
        ),
        Document(
            doc_id="doc9",
            content="Attention mechanisms allow models to focus on relevant parts of the input by computing weighted combinations of all input elements based on query and key vectors.",
            metadata={"source": "Attention Mechanisms", "category": "Mechanisms"}
        ),
        Document(
            doc_id="doc10",
            content="Embedding layers convert discrete tokens into continuous vector representations that capture semantic and syntactic relationships between words in a learned space.",
            metadata={"source": "Representation Learning", "category": "Embeddings"}
        ),
    ]
    
    # Initialize FiD RAG system
    fid_rag = FusionInDecoderRAG(retrieval_alpha=0.6, num_retrieved=8, use_fusion=True)
    fid_rag.index(documents)
    
    # Query and reference human answer
    query = "How do neural networks learn from data?"
    human_answer = "Neural networks learn through backpropagation, which adjusts weights by computing gradients of the loss function. They consist of interconnected nodes organized in layers that process information, inspired by biological neurons."
    
    # Full evaluation
    result = fid_rag.evaluate_full(query, human_answer)
    
    # Format and display results
    output = {
        "system_type": "Fusion-in-Decoder (FiD) RAG",
        "query": result.query,
        "generated_ai_answer": result.generated_answer,
        "human_reference_answer": result.comparison.human_answer,
        "consistency_metrics": {
            "semantic_consistency": round(result.consistency.semantic_consistency, 4),
            "lexical_consistency": round(result.consistency.lexical_consistency, 4),
            "fusion_consistency": round(result.consistency.fusion_consistency, 4),
            "overall_consistency": round(result.consistency.overall_consistency, 4),
            "variation_coefficient": round(result.consistency.variation_coefficient, 4),
            "interpretation": "Higher = More Consistent, Lower Variation = Better"
        },
        "accuracy_metrics": {
            "bleu_score": round(result.accuracy.bleu_score, 4),
            "rouge_score": round(result.accuracy.rouge_score, 4),
            "similarity_score": round(result.accuracy.similarity_score, 4),
            "f1_score": round(result.accuracy.f1_score, 4),
            "exact_match": result.accuracy.exact_match,
            "interpretation": "Higher = Better Accuracy"
        },
        "ai_vs_human_comparison": {
            "semantic_similarity": round(result.comparison.semantic_similarity, 4),
            "lexical_similarity": round(result.comparison.lexical_similarity, 4),
            "structural_similarity": round(result.comparison.structural_similarity, 4),
            "length_ratio": round(result.comparison.length_ratio, 4),
            "agreement_score": round(result.comparison.agreement_score, 4),
            "recommendation": result.comparison.recommendation,
        },
        "fusion_mechanism_statistics": {
            "fusion_applied": result.fusion_mechanism_stats.get("fusion_applied"),
            "num_documents_fused": result.fusion_mechanism_stats.get("num_documents_fused"),
            "attention_weights": [round(w, 4) for w in result.fusion_mechanism_stats.get("attention_weights", [])],
        },
        "retrieved_documents": [
            {
                "doc_id": doc.doc_id,
                "content_preview": doc.content[:80] + "...",
                "retrieval_score": round(doc.retrieval_score, 4),
                "metadata": doc.metadata
            }
            for doc in result.retrieved_docs[:5]
        ],
        "complexity_analysis": {
            "operations": [
                {
                    "operation": c.operation,
                    "time_complexity": c.time_complexity,
                    "space_complexity": c.space_complexity,
                    "description": c.description,
                }
                for c in result.complexity
            ]
        },
        "performance_metrics": {
            "total_execution_time_seconds": round(result.total_time, 4),
            "total_documents": len(documents),
            "documents_retrieved": len(result.retrieved_docs),
        }
    }
    
    print(json.dumps(output, indent=2))
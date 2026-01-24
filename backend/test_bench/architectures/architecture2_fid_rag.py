"""Architecture 2: Fusion-in-Decoder (FiD) RAG - Debug Version"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from architectures.base import BaseRAGArchitecture
from bench_core.document import Document
from bench_core.result import ArchitectureResult

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.answer_generator import generate_answer


class FiDRAGArchitecture(BaseRAGArchitecture):
    """Fusion-in-Decoder RAG Architecture"""

    def __init__(self):
        super().__init__("Fusion-in-Decoder (FiD) RAG Architecture")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_embeddings = {}
        self.bm25_index = None
        self.fusion_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def train(self, documents: List[Document]):
        """Train on documents"""
        print(f"[FiD] Training with {len(documents)} documents")
        self.documents = documents

        # Create vector embeddings
        texts = [doc.content for doc in documents]
        print(f"[FiD] Creating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        for doc, emb in zip(documents, embeddings):
            self.vector_embeddings[doc.doc_id] = emb

        print(f"[FiD] Created {len(self.vector_embeddings)} vector embeddings")

        # Create BM25 index
        tokenized_corpus = [doc.content.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"[FiD] BM25 index created")

        self.is_trained = True
        print(f"[FiD] Training complete")

    def query(self, query: str) -> ArchitectureResult:
        """Process query using FiD approach"""
        print(f"\n[FiD] Processing query: '{query}'")
        start_time = time.time()

        try:
            # Check if trained
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")

            # Retrieve documents using hybrid search
            print(f"[FiD] Creating query embedding...")
            query_embedding = self.embedding_model.encode(query)

            # Vector similarity
            print(f"[FiD] Calculating vector similarities...")
            vector_scores = {}
            for doc_id, doc_emb in self.vector_embeddings.items():
                similarity = float(
                    np.dot(query_embedding, doc_emb)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-8)
                )
                vector_scores[doc_id] = similarity

            print(f"[FiD] Vector scores calculated for {len(vector_scores)} documents")
            print(
                f"[FiD] Top 3 vector scores: {sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)[:3]}"
            )

            # BM25 scores
            print(f"[FiD] Calculating BM25 scores...")
            query_tokens = query.lower().split()
            print(f"[FiD] Query tokens: {query_tokens}")
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            print(
                f"[FiD] BM25 scores: min={np.min(bm25_scores):.4f}, max={np.max(bm25_scores):.4f}, mean={np.mean(bm25_scores):.4f}"
            )

            # Combine scores
            print(f"[FiD] Combining scores (70% vector, 30% BM25)...")
            combined_scores = {}
            for idx, doc in enumerate(self.documents):
                vector_score = vector_scores.get(doc.doc_id, 0.0)
                bm25_score = float(bm25_scores[idx])
                bm25_norm = bm25_score / (bm25_score + 1.0) if bm25_score > 0 else 0.0
                combined_scores[doc.doc_id] = 0.7 * vector_score + 0.3 * bm25_norm

            # Get top documents
            top_docs = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )[:10]
            print(
                f"[FiD] Top 10 combined scores: {[(doc_id, f'{score:.4f}') for doc_id, score in top_docs]}"
            )

            # Create a dict for quick lookup
            doc_dict = {doc.doc_id: doc for doc in self.documents}
            # Get documents in the order of top_docs
            retrieved_docs = [doc_dict[d[0]] for d in top_docs if d[0] in doc_dict]
            print(f"[FiD] Retrieved {len(retrieved_docs)} documents")

            # FiD: Encode query with each document separately, then fuse
            if retrieved_docs:
                print(
                    f"[FiD] Applying Fusion-in-Decoder on top {min(5, len(retrieved_docs))} documents..."
                )
                # Encode query-document pairs
                query_doc_pairs = [
                    f"{query} [SEP] {doc.content}" for doc in retrieved_docs[:5]
                ]
                print(f"[FiD] Encoding {len(query_doc_pairs)} query-document pairs...")
                fused_embeddings = self.fusion_encoder.encode(
                    query_doc_pairs, show_progress_bar=False
                )

                # Aggregate (mean pooling)
                fused_embedding = np.mean(fused_embeddings, axis=0)
                print(f"[FiD] Created fused embedding of shape {fused_embedding.shape}")

                # Find most similar document to fused representation
                doc_embeddings = np.array(
                    [self.vector_embeddings[doc.doc_id] for doc in retrieved_docs[:5]]
                )
                similarities = np.dot(doc_embeddings, fused_embedding) / (
                    np.linalg.norm(doc_embeddings, axis=1)
                    * np.linalg.norm(fused_embedding)
                    + 1e-8
                )

                print(f"[FiD] Fusion similarities: {similarities}")

                # Get top documents based on fusion
                top_indices = np.argsort(similarities)[::-1][:3]
                final_docs = [retrieved_docs[i] for i in top_indices]
                final_texts = [doc.content for doc in final_docs]

                print(
                    f"[FiD] Selected {len(final_docs)} final documents for answer generation"
                )
                print(f"[FiD] Document IDs: {[doc.doc_id for doc in final_docs]}")
                print(
                    f"[FiD] Document previews: {[text[:100] + '...' for text in final_texts]}"
                )

                # Generate focused answer
                print(f"[FiD] Generating answer...")
                output = generate_answer(query, final_texts, max_length=500)
                print(f"[FiD] Generated output length: {len(output)} characters")
                print(f"[FiD] Output preview: {output[:200]}...")

                # Calculate metrics
                latency = time.time() - start_time
                max_similarity = (
                    float(np.max(similarities)) if len(similarities) > 0 else 0.0
                )
                confidence_score = max_similarity
                evidence_score = len(final_docs) / 5.0
                accuracy = (confidence_score + evidence_score) / 2.0
                average_accuracy = accuracy

                print(
                    f"[FiD] Metrics - Latency: {latency:.2f}s, Accuracy: {accuracy:.4f}, Confidence: {confidence_score:.4f}"
                )

            else:
                print(f"[FiD] No relevant documents found!")
                output = "No relevant documents found."
                latency = time.time() - start_time
                confidence_score = 0.0
                evidence_score = 0.0
                accuracy = 0.0
                average_accuracy = 0.0

            result = ArchitectureResult(
                architect_name=self.name,
                input_query=query,
                output=output,
                latency=latency,
                accuracy=accuracy,
                confidence_score=confidence_score,
                evidence_score=evidence_score,
                average_accuracy=average_accuracy,
            )

            print(f"[FiD] Query complete. Returning result.")
            return result

        except Exception as e:
            print(f"[FiD] ERROR: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()

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
                error=str(e),
            )


import time
import numpy as np
from typing import List, Set
from core.contracts.query import Query
from core.contracts.retrieval_plan import RetrievalPlan
from core.contracts.retrieved_chunk import RetrievedChunk
from core.contracts.retrieve_result import RetrieveResult
from data.embedding_engine import EmbeddingEngine
from data.faiss_indexer import FAISSIndexer
from logger import Logger


class RetrievalEngine:
    """Main retrieval engine using per-topic FAISS"""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        faiss_indexer: FAISSIndexer,
        chunks: List,
    ):
        self.embedding_engine = embedding_engine
        self.faiss_indexer = faiss_indexer
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.logger = Logger().get_logger("Retrieval")

    def retrieve(self, query: Query, plan: RetrievalPlan) -> RetrieveResult:
        """Retrieve chunks based on plan"""

        start_time = time.time()

        # Embed query
        query_embedding = self.embedding_engine.embed_query(query.text)

        # Search using FAISS
        if plan.use_ann:
            results = self.faiss_indexer.search(
                query_embedding, list(plan.topic_ids), plan.max_chunks
            )
        else:
            # Fallback to exhaustive search (not implemented here)
            self.logger.warning("Non-ANN retrieval not implemented, using ANN")
            results = self.faiss_indexer.search(
                query_embedding, list(plan.topic_ids), plan.max_chunks
            )

        # Convert to RetrievedChunk objects
        chunks = []
        for chunk_id, similarity, topic_id in results:
            if chunk_id not in self.chunks:
                self.logger.warning(f"Chunk {chunk_id} not found in chunk store")
                continue

            chunk_data = self.chunks[chunk_id]

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk_data.text,
                topic_id=topic_id,
                similarity_score=similarity,
                chunk_index=chunk_data.chunk_index,
                source_doc=chunk_data.source_doc,
            )
            chunks.append(retrieved_chunk)

        retrieval_time_ms = (time.time() - start_time) * 1000

        result = RetrieveResult(
            chunks=tuple(chunks),
            from_cache=False,
            total_searched=sum(
                self.faiss_indexer.indexes[tid].ntotal
                for tid in plan.topic_ids
                if tid in self.faiss_indexer.indexes
            ),
            retrieval_time_ms=retrieval_time_ms,
        )

        self.logger.info(
            f"Retrieved {len(chunks)} chunks in {retrieval_time_ms:.1f}ms "
            f"(searched {result.total_searched} vectors)"
        )

        return result

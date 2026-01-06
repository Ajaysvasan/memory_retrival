from typing import List

import numpy as np
from topic_router import TopicRouter

from config.thresholds import Chunk, Query
from ingest.embedder import Embedder
from ingest.indexer import FAISSIndexer


class Retriever:
    """Main retrieval orchestrator."""

    def __init__(
        self, indexer: FAISSIndexer, embedder: Embedder, topic_router: TopicRouter
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.topic_router = topic_router

    def retrieve(self, query: Query, k: int = 10) -> List[Chunk]:
        """Execute retrieval with adaptive K."""
        query_emb = self.embedder.embed_query(query.text)

        # Get initial results
        chunks = self.indexer.search(query_emb, k=k)

        # Early cutoff if high confidence
        if chunks and self._high_confidence(chunks[0], query_emb):
            return chunks[: k // 2]

        return chunks

    def _high_confidence(self, chunk: Chunk, query_emb: np.ndarray) -> bool:
        """Check if top result has high confidence."""
        if chunk.embedding is None:
            return False
        similarity = np.dot(chunk.embedding, query_emb)
        return similarity > 0.85

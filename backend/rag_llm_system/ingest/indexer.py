import pickle
from typing import List

import faiss
import numpy as np

from config.thresholds import CONFIG, Chunk


class FAISSIndexer:
    """FAISS HNSW index builder."""

    def __init__(self, dimension: int = CONFIG.EMBEDDING_DIM):
        self.dimension = dimension
        self.index = None
        self.chunk_map = {}

    def build_index(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Build HNSW index."""
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, CONFIG.HNSW_M)
        self.index.hnsw.efConstruction = CONFIG.HNSW_EF_CONSTRUCTION

        # Add vectors
        self.index.add(embeddings.astype("float32"))

        # Map indices to chunks
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Chunk]:
        """Search with adaptive efSearch."""
        self.index.hnsw.efSearch = max(k * 2, CONFIG.HNSW_EF_SEARCH)

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), k
        )

        return [self.chunk_map[idx] for idx in indices[0] if idx in self.chunk_map]

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.map", "wb") as f:
            pickle.dump(self.chunk_map, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.map", "rb") as f:
            self.chunk_map = pickle.load(f)

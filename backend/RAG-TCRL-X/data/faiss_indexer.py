import numpy as np
import faiss
from typing import List, Dict
from pathlib import Path
from logger import Logger
from config import Config


class FAISSIndexer:
    """Per-topic FAISS HNSW indexing"""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.logger = Logger().get_logger("FAISS")
        self.indexes: Dict[int, faiss.IndexHNSWFlat] = {}
        self.topic_chunk_maps: Dict[int, List[int]] = {}

    def build_indexes(
        self,
        embeddings: np.ndarray,
        topic_assignments: np.ndarray,
        chunk_ids: List[int],
    ):
        """Build per-topic FAISS HNSW indexes"""

        if len(embeddings) != len(topic_assignments) != len(chunk_ids):
            raise ValueError("Embeddings, topics, and chunk_ids must have same length")

        self.logger.info("Building per-topic FAISS indexes...")

        # Group by topic
        topic_groups: Dict[int, List[int]] = {}
        for idx, topic_id in enumerate(topic_assignments):
            topic_id = int(topic_id)
            if topic_id not in topic_groups:
                topic_groups[topic_id] = []
            topic_groups[topic_id].append(idx)

        # Build index for each topic
        for topic_id, indices in topic_groups.items():
            if not indices:
                continue

            # Get embeddings for this topic
            topic_embeddings = embeddings[indices].astype("float32")
            topic_chunk_ids = [chunk_ids[i] for i in indices]

            # Create HNSW index
            index = faiss.IndexHNSWFlat(self.embedding_dim, Config.FAISS_M)
            index.hnsw.efConstruction = Config.FAISS_EF_CONSTRUCTION

            # Add vectors
            index.add(topic_embeddings)

            self.indexes[topic_id] = index
            self.topic_chunk_maps[topic_id] = topic_chunk_ids

            self.logger.info(
                f"Topic {topic_id}: built HNSW index with {len(indices)} vectors"
            )

        self.logger.info(f"Built {len(self.indexes)} topic indexes")

    def search(
        self, query_embedding: np.ndarray, topic_ids: List[int], k: int
    ) -> List[tuple]:
        """Search across specified topics

        Returns: List of (chunk_id, similarity_score, topic_id) tuples
        """

        if not self.indexes:
            raise RuntimeError("No indexes built")

        query_vector = query_embedding.reshape(1, -1).astype("float32")

        all_results = []

        for topic_id in topic_ids:
            if topic_id not in self.indexes:
                self.logger.warning(f"Topic {topic_id} not in indexes, skipping")
                continue

            index = self.indexes[topic_id]
            chunk_map = self.topic_chunk_maps[topic_id]

            # Search in this topic's index
            # FAISS returns distances, we convert to similarities
            k_search = min(k, index.ntotal)
            distances, indices = index.search(query_vector, k_search)

            # Convert to results
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(chunk_map):
                    continue

                chunk_id = chunk_map[idx]
                # Convert L2 distance to cosine similarity (embeddings are normalized)
                similarity = 1.0 / (1.0 + dist)

                all_results.append((chunk_id, float(similarity), topic_id))

        # Sort by similarity descending
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return all_results[:k]

    def save_indexes(self, directory: Path):
        """Save all indexes to directory"""
        directory.mkdir(parents=True, exist_ok=True)

        for topic_id, index in self.indexes.items():
            index_path = directory / f"topic_{topic_id}.index"
            faiss.write_index(index, str(index_path))

        # Save chunk maps
        import pickle

        map_path = directory / "chunk_maps.pkl"
        with open(map_path, "wb") as f:
            pickle.dump(self.topic_chunk_maps, f)

        self.logger.info(f"Saved {len(self.indexes)} indexes to {directory}")

    def load_indexes(self, directory: Path):
        """Load all indexes from directory"""
        if not directory.exists():
            raise FileNotFoundError(f"Index directory not found: {directory}")

        # Load chunk maps
        import pickle

        map_path = directory / "chunk_maps.pkl"
        with open(map_path, "rb") as f:
            self.topic_chunk_maps = pickle.load(f)

        # Load indexes
        for topic_id in self.topic_chunk_maps.keys():
            index_path = directory / f"topic_{topic_id}.index"
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                self.indexes[topic_id] = index

        self.logger.info(f"Loaded {len(self.indexes)} indexes from {directory}")

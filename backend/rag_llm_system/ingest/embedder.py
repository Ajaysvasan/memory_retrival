# ingest/embedder.py
from typing import List

import numpy as np
import torch

from config.thresholds import CONFIG


class Embedder:
    """Single embedding model for all operations."""

    def __init__(self, model_name: str = CONFIG.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = CONFIG.EMBEDDING_DIM

    def embed_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Batch embedding with normalization."""
        embeddings = self.model.encode(
            texts,
            batch_size=CONFIG.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Single query embedding."""
        return self.embed_batch([text])[0]

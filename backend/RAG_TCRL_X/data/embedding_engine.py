from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import Config
from logger import Logger


class EmbeddingEngine:
    """Handles text embedding with hardware safety"""

    def __init__(self):
        self.logger = Logger().get_logger("Embedding")

        # Setup device with hardware safety
        self.device = self._setup_device()

        # Load embedding model
        self.logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(
            Config.EMBEDDING_MODEL, device=str(self.device)
        )

        self.logger.info(f"Embedding engine initialized on {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup compute device with hardware safety"""
        if Config.FORCE_CPU:
            self.logger.info("CPU mode forced by configuration")
            return torch.device("cpu")

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

        # Test CUDA usability
        try:
            test_tensor = torch.zeros(Config.GPU_TEST_TENSOR_SIZE).cuda()
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()

            self.logger.info("GPU test passed, using CUDA")
            return torch.device("cuda")

        except Exception as e:
            self.logger.warning(f"GPU test failed: {e}, falling back to CPU")
            return torch.device("cpu")

    def embed_chunks(self, chunks: List) -> np.ndarray:
        """Embed list of chunks"""
        if not chunks:
            raise ValueError("Cannot embed empty chunk list")

        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed list of texts"""
        if not texts:
            raise ValueError("Cannot embed empty text list")

        self.logger.info(f"Embedding {len(texts)} texts...")

        try:
            # Encode in batches
            embeddings = self.model.encode(
                texts,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            self.logger.info(f"Generated embeddings: shape={embeddings.shape}")
            return embeddings

        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                self.logger.warning(f"GPU error during embedding: {e}, retrying on CPU")

                # Move model to CPU
                self.model = self.model.to("cpu")
                self.device = torch.device("cpu")

                # Retry on CPU
                embeddings = self.model.encode(
                    texts,
                    batch_size=Config.BATCH_SIZE,
                    show_progress_bar=len(texts) > 100,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                return embeddings
            else:
                raise

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed single query"""
        embeddings = self.embed_texts([query_text])
        return embeddings[0]

    def save_embeddings(self, embeddings: np.ndarray, filepath: Path):
        """Save embeddings to disk"""
        np.save(filepath, embeddings)
        self.logger.info(f"Saved embeddings to {filepath}")

    def load_embeddings(self, filepath: Path) -> np.ndarray:
        """Load embeddings from disk"""
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        embeddings = np.load(filepath)
        self.logger.info(f"Loaded embeddings from {filepath}: shape={embeddings.shape}")
        return embeddings

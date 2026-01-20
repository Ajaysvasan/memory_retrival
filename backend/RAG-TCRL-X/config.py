import os
from pathlib import Path
from typing import List, Optional


class Config:
    """System-wide configuration with validation"""

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DATASET_DIR = DATA_DIR / "datasets"  # Changed to directory for multiple files
    EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
    TOPICS_PATH = DATA_DIR / "topic_centroids.npy"
    FAISS_DIR = DATA_DIR / "faiss_indexes"
    RL_MODEL_PATH = DATA_DIR / "rl_agent.pt"
    CACHE_PATH = DATA_DIR / "cache.pkl"
    BELIEFS_PATH = DATA_DIR / "beliefs.pkl"

    # Supported file formats
    SUPPORTED_FORMATS = [".txt", ".pdf", ".docx", ".doc"]

    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    BATCH_SIZE = 32

    # Chunking
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    MIN_CHUNK_LENGTH = 50

    # Topic modeling
    NUM_TOPICS = 10
    TOPIC_MIN_SAMPLES = 5

    # FAISS
    FAISS_NPROBE = 10
    FAISS_M = 16
    FAISS_EF_CONSTRUCTION = 200

    # Retrieval
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7

    # Validation
    EVIDENCE_THRESHOLD = 0.75
    MIN_EVIDENCE_RATIO = 0.6
    CONTRADICTION_THRESHOLD = 0.85

    # Cache
    CACHE_TTL_SECONDS = 3600
    CACHE_MIN_FREQUENCY = 2
    CACHE_ADMISSION_THRESHOLD = 0.8

    # Belief system
    INITIAL_CONFIDENCE = 0.8
    CONFIDENCE_DECAY = 0.1
    MIN_CONFIDENCE = 0.3

    # RL
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.99
    RL_EPSILON = 0.1
    RL_ALPHA_CORRECT = 1.0
    RL_BETA_LATENCY = 0.1
    RL_GAMMA_MEMORY = 0.05
    RL_DELTA_HALLUCINATION = 2.0
    RL_REWARD_CLIP = 10.0

    # Hardware
    FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
    GPU_TEST_TENSOR_SIZE = 100

    @classmethod
    def validate(cls):
        """Validate configuration and create necessary directories"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        cls.FAISS_DIR.mkdir(parents=True, exist_ok=True)

        # Check if dataset directory exists and has files
        if not cls.DATASET_DIR.exists():
            raise RuntimeError(
                f"Dataset directory does not exist: {cls.DATASET_DIR}\n"
                f"Create the directory and add your dataset files (.txt, .pdf, .docx)"
            )

        # Find all supported files
        dataset_files = cls.get_dataset_files()

        if not dataset_files:
            raise RuntimeError(
                f"No dataset files found in {cls.DATASET_DIR}\n"
                f"Supported formats: {', '.join(cls.SUPPORTED_FORMATS)}\n"
                f"Please add at least one file to the datasets directory."
            )

        if cls.NUM_TOPICS < 2:
            raise ValueError("NUM_TOPICS must be >= 2")

        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            raise ValueError("CHUNK_SIZE must be > CHUNK_OVERLAP")

        if cls.TOP_K < 1:
            raise ValueError("TOP_K must be >= 1")

    @classmethod
    def get_dataset_files(cls) -> List[Path]:
        """Get all dataset files from the dataset directory"""
        if not cls.DATASET_DIR.exists():
            return []

        dataset_files = []
        for ext in cls.SUPPORTED_FORMATS:
            dataset_files.extend(cls.DATASET_DIR.glob(f"*{ext}"))

        return sorted(dataset_files)

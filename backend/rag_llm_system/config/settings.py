from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemConfig:
    # Chunking
    CHUNK_SIZE_MIN: int = 200
    CHUNK_SIZE_MAX: int = 400
    CHUNK_OVERLAP_WORDS: int = 50

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    BATCH_SIZE: int = 32

    # FAISS
    FAISS_INDEX_TYPE: str = "HNSW"
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 50

    # Cache
    L1_CACHE_SIZE: int = 1000
    L2_CACHE_SIZE: int = 10000
    CACHE_TTL_SECONDS: int = 3600

    # Memory
    STM_MAX_TURNS: int = 5
    WM_MAX_SIZE: int = 2048
    EM_MAX_EPISODES: int = 100

    # Paths
    DATA_DIR: Path = Path("data")
    RAW_DIR: Path = DATA_DIR / "raw"
    INDEX_DIR: Path = DATA_DIR / "indices"
    CACHE_DIR: Path = DATA_DIR / "cache"

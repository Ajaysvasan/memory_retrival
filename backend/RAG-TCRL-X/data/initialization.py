import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import Config
from data.embedding_engine import EmbeddingEngine
from data.faiss_indexer import FAISSIndexer
from data.ingestion_engine import Chunk, IngestionEngine
from data.topic_modeler import TopicModeler
from logger import Logger


class SystemInitializer:
    """Handles full system initialization and data pipeline"""

    def __init__(self):
        self.logger = Logger().get_logger("Initializer")
        self.ingestion = IngestionEngine()
        self.embedding_engine = EmbeddingEngine()
        self.topic_modeler = TopicModeler()
        self.faiss_indexer = FAISSIndexer(Config.EMBEDDING_DIM)

    def initialize(self) -> Tuple[List[Chunk], EmbeddingEngine, FAISSIndexer]:
        """Run full initialization pipeline"""

        self.logger.info("=" * 80)
        self.logger.info("STARTING SYSTEM INITIALIZATION")
        self.logger.info("=" * 80)

        documents = self.ingestion.load_all_documents()
        self.logger.info(f"Loaded {len(documents)} total documents")

        chunks = self.ingestion.chunk_documents(documents)

        if not chunks:
            raise RuntimeError(
                "CRITICAL: Chunking produced 0 chunks - system invariant violated"
            )

        stats = self.ingestion.get_statistics(chunks)
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"  Total chunks: {stats['total_chunks']}")

        if stats["corpora"]:
            self.logger.info("  Chunks by corpus:")
            for corpus, count in sorted(stats["corpora"].items()):
                self.logger.info(f"    - {corpus}: {count} chunks")
                if corpus in stats["topics"]:
                    for topic, topic_count in sorted(stats["topics"][corpus].items()):
                        self.logger.info(f"      └─ {topic}: {topic_count} chunks")

        self.logger.info("  Chunks by file:")
        for filename, count in sorted(stats["files"].items())[:10]:
            self.logger.info(f"    - {filename}: {count} chunks")
        if len(stats["files"]) > 10:
            self.logger.info(f"    ... and {len(stats['files']) - 10} more files")

        self.logger.info("  Chunks by format:")
        for file_format, count in sorted(stats["formats"].items()):
            self.logger.info(f"    - {file_format}: {count} chunks")

        chunks = self.ingestion.deduplicate_chunks(chunks)

        embeddings = self.embedding_engine.embed_chunks(chunks)

        if len(embeddings) == 0:
            raise RuntimeError(
                "CRITICAL: Embedding produced 0 embeddings - system invariant violated"
            )

        self.embedding_engine.save_embeddings(embeddings, Config.EMBEDDINGS_PATH)

        centroids = self.topic_modeler.train(embeddings)

        if centroids is None or len(centroids) == 0:
            raise RuntimeError(
                "CRITICAL: Topic model training produced 0 centroids - system invariant violated"
            )

        topic_assignments = self.topic_modeler.assign_topics(embeddings)

        self.topic_modeler.save_centroids(Config.TOPICS_PATH)

        self.topic_modeler.get_topic_distribution(topic_assignments)

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.faiss_indexer.build_indexes(embeddings, topic_assignments, chunk_ids)

        self.faiss_indexer.save_indexes(Config.FAISS_DIR)

        self._save_chunks(chunks)

        self.logger.info("=" * 80)
        self.logger.info("INITIALIZATION COMPLETE")
        self.logger.info("=" * 80)

        return chunks, self.embedding_engine, self.faiss_indexer

    def load_existing(self) -> Tuple[List[Chunk], EmbeddingEngine, FAISSIndexer]:
        """Load existing initialized system"""

        self.logger.info("=" * 80)
        self.logger.info("LOADING EXISTING SYSTEM")
        self.logger.info("=" * 80)

        chunks = self._load_chunks()
        self.logger.info(f"Loaded {len(chunks)} chunks")

        stats = self.ingestion.get_statistics(chunks)
        self.logger.info("Loaded Dataset Statistics:")
        self.logger.info(f"  Total chunks: {stats['total_chunks']}")

        if stats["corpora"]:
            self.logger.info("  Chunks by corpus:")
            for corpus, count in sorted(stats["corpora"].items()):
                self.logger.info(f"    - {corpus}: {count} chunks")

        self.logger.info("  Chunks by file:")
        for filename, count in sorted(stats["files"].items())[:10]:
            self.logger.info(f"    - {filename}: {count} chunks")
        if len(stats["files"]) > 10:
            self.logger.info(f"    ... and {len(stats['files']) - 10} more files")

        embeddings = self.embedding_engine.load_embeddings(Config.EMBEDDINGS_PATH)
        self.logger.info(f"Loaded embeddings: shape={embeddings.shape}")

        self.topic_modeler.load_centroids(Config.TOPICS_PATH)

        self.faiss_indexer.load_indexes(Config.FAISS_DIR)

        self.logger.info("=" * 80)
        self.logger.info("SYSTEM LOADED SUCCESSFULLY")
        self.logger.info("=" * 80)

        return chunks, self.embedding_engine, self.faiss_indexer

    def _save_chunks(self, chunks: List[Chunk]):
        """Save chunk metadata"""
        chunks_path = Config.DATA_DIR / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        self.logger.info(f"Saved {len(chunks)} chunks metadata")

    def _load_chunks(self) -> List[Chunk]:
        """Load chunk metadata"""
        chunks_path = Config.DATA_DIR / "chunks.pkl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        return chunks


class IntegrityValidator:
    """Validates system integrity on load"""

    def __init__(self):
        self.logger = Logger().get_logger("IntegrityValidator")

    def validate(
        self,
        chunks: List[Chunk],
        embedding_engine: EmbeddingEngine,
        faiss_indexer: FAISSIndexer,
    ):
        """Validate loaded system integrity"""

        self.logger.info("Validating system integrity...")

        if not chunks:
            raise RuntimeError("INTEGRITY VIOLATION: No chunks loaded")

        if not Config.EMBEDDINGS_PATH.exists():
            raise RuntimeError("INTEGRITY VIOLATION: Embeddings file missing")

        embeddings = np.load(Config.EMBEDDINGS_PATH)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"INTEGRITY VIOLATION: Embedding count ({len(embeddings)}) "
                f"!= chunk count ({len(chunks)})"
            )

        if not Config.TOPICS_PATH.exists():
            raise RuntimeError("INTEGRITY VIOLATION: Topic centroids missing")

        if not faiss_indexer.indexes:
            raise RuntimeError("INTEGRITY VIOLATION: No FAISS indexes loaded")

        total_indexed = sum(idx.ntotal for idx in faiss_indexer.indexes.values())
        if total_indexed != len(chunks):
            raise RuntimeError(
                f"INTEGRITY VIOLATION: Total indexed vectors ({total_indexed}) "
                f"!= chunk count ({len(chunks)})"
            )

        self.logger.info("✓ System integrity validated")

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from config import Config
from data.document_loader import Document, DocumentLoaderFactory
from logger import Logger


class Chunk:
    """Text chunk with metadata"""

    def __init__(
        self,
        text: str,
        chunk_id: int,
        source_doc: str,
        chunk_index: int,
        metadata: dict = None,
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.source_doc = source_doc
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class IngestionEngine:
    """Handles document loading, chunking, and deduplication from multiple files"""

    def __init__(self):
        self.logger = Logger().get_logger("Ingestion")
        self.loader_factory = DocumentLoaderFactory()

    def load_all_documents(self) -> List[Document]:
        """Load all documents from dataset directory recursively"""

        dataset_files = Config.get_dataset_files()

        if not dataset_files:
            raise RuntimeError(
                f"No dataset files found in {Config.DATASET_DIR}\n"
                f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
            )

        self.logger.info(
            f"Discovering dataset files recursively from {Config.DATASET_DIR}"
        )
        self.logger.info(f"Found {len(dataset_files)} dataset files")

        corpus_stats = defaultdict(int)
        topic_stats = defaultdict(lambda: defaultdict(int))

        for filepath in dataset_files:
            try:
                parts = filepath.relative_to(Config.DATASET_DIR).parts
                if len(parts) >= 1:
                    corpus = parts[0]
                    corpus_stats[corpus] += 1
                    if len(parts) >= 2:
                        topic = parts[1]
                        topic_stats[corpus][topic] += 1
            except ValueError:
                pass

        self.logger.info("Dataset file distribution:")
        for corpus, count in sorted(corpus_stats.items()):
            self.logger.info(f"  Corpus '{corpus}': {count} files")
            if corpus in topic_stats:
                for topic, topic_count in sorted(topic_stats[corpus].items()):
                    self.logger.info(f"    Topic '{topic}': {topic_count} files")

        all_documents = []
        failed_files = []
        corpus_doc_counts = defaultdict(int)

        for filepath in dataset_files:
            try:
                self.logger.info(
                    f"Processing: {filepath.relative_to(Config.DATASET_DIR)}"
                )
                documents = self.loader_factory.load_file(filepath)
                all_documents.extend(documents)

                if documents and "corpus" in documents[0].metadata:
                    corpus = documents[0].metadata["corpus"]
                    corpus_doc_counts[corpus] += len(documents)

                self.logger.info(
                    f"✓ Loaded {len(documents)} documents from {filepath.name}"
                )

            except Exception as e:
                self.logger.error(f"✗ Failed to load {filepath.name}: {e}")
                failed_files.append((filepath.name, str(e)))

        if not all_documents:
            error_msg = "No documents could be loaded from any file"
            if failed_files:
                error_msg += "\n\nFailed files:\n"
                for filename, error in failed_files:
                    error_msg += f"  - {filename}: {error}\n"
            raise RuntimeError(error_msg)

        self.logger.info(f"Total documents loaded: {len(all_documents)}")
        self.logger.info("Documents per corpus:")
        for corpus, count in sorted(corpus_doc_counts.items()):
            self.logger.info(f"  {corpus}: {count} documents")

        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} files:")
            for filename, error in failed_files:
                self.logger.warning(f"  - {filename}: {error}")

        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents into smaller pieces"""
        self.logger.info("Chunking documents...")

        all_chunks = []
        chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            doc_chunks = self._chunk_text(doc.text, doc.source, chunk_id, doc.metadata)

            if not doc_chunks:
                self.logger.warning(
                    f"Document {doc_idx} from {doc.metadata.get('filename', 'unknown')} "
                    f"produced 0 chunks, using full text fallback"
                )
                chunk = Chunk(
                    text=doc.text,
                    chunk_id=chunk_id,
                    source_doc=doc.source,
                    chunk_index=0,
                    metadata=doc.metadata,
                )
                doc_chunks = [chunk]
                chunk_id += 1
            else:
                chunk_id += len(doc_chunks)

            all_chunks.extend(doc_chunks)

        self.logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents"
        )

        if not all_chunks:
            raise RuntimeError(
                "Chunking produced 0 chunks - this violates system invariants"
            )

        return all_chunks

    def _chunk_text(
        self, text: str, source: str, start_id: int, metadata: dict
    ) -> List[Chunk]:
        """Chunk single text using sliding window"""
        if len(text) < Config.MIN_CHUNK_LENGTH:
            return []

        chunks = []
        chunk_index = 0

        sentences = self._split_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > Config.CHUNK_SIZE and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= Config.MIN_CHUNK_LENGTH:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=start_id + chunk_index,
                        source_doc=source,
                        chunk_index=chunk_index,
                        metadata=metadata.copy() if metadata else {},
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, Config.CHUNK_OVERLAP
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= Config.MIN_CHUNK_LENGTH:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=start_id + chunk_index,
                    source_doc=source,
                    chunk_index=chunk_index,
                    metadata=metadata.copy() if metadata else {},
                )
                chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self, sentences: List[str], target_length: int
    ) -> List[str]:
        """Get sentences for overlap"""
        overlap = []
        length = 0

        for sentence in reversed(sentences):
            if length + len(sentence) > target_length:
                break
            overlap.insert(0, sentence)
            length += len(sentence)

        return overlap

    def deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks"""
        self.logger.info("Deduplicating chunks...")

        seen_texts = set()
        unique_chunks = []

        for chunk in chunks:
            normalized = chunk.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)

        removed = len(chunks) - len(unique_chunks)
        self.logger.info(
            f"Removed {removed} duplicate chunks, {len(unique_chunks)} remain"
        )

        return unique_chunks

    def get_statistics(self, chunks: List[Chunk]) -> dict:
        """Get statistics about loaded chunks"""
        stats = {
            "total_chunks": len(chunks),
            "files": {},
            "formats": {},
            "corpora": {},
            "topics": defaultdict(lambda: defaultdict(int)),
        }

        for chunk in chunks:
            filename = chunk.metadata.get("filename", "unknown")
            file_format = chunk.metadata.get("format", "unknown")
            corpus = chunk.metadata.get("corpus", "unknown")
            topic_hint = chunk.metadata.get("topic_hint", "unknown")

            if filename not in stats["files"]:
                stats["files"][filename] = 0
            stats["files"][filename] += 1

            if file_format not in stats["formats"]:
                stats["formats"][file_format] = 0
            stats["formats"][file_format] += 1

            if corpus not in stats["corpora"]:
                stats["corpora"][corpus] = 0
            stats["corpora"][corpus] += 1

            stats["topics"][corpus][topic_hint] += 1

        return stats

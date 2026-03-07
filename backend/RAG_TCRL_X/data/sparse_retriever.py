import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from logger import Logger


@dataclass
class SparseHit:
    chunk_id: int
    score: float
    topic_id: int


class RankBM25SparseBackend:
    """In-memory BM25 backend using rank-bm25."""

    def __init__(self, chunks: Sequence, chunk_topic_map: Dict[int, int]):
        self.logger = Logger().get_logger("SparseRankBM25")
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_topic_map = chunk_topic_map
        self.tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, topic_ids: Sequence[int], k: int) -> List[SparseHit]:
        query_tokens = query.lower().split()
        scores = self.index.get_scores(query_tokens)

        allowed_topics = set(topic_ids)
        hits: List[SparseHit] = []

        for idx, score in enumerate(scores):
            chunk_id = self.chunk_ids[idx]
            topic_id = self.chunk_topic_map.get(chunk_id, -1)
            if topic_id not in allowed_topics:
                continue

            # Normalize sparse score to [0,1)-ish range for compatibility.
            normalized = float(score / (score + 1.0)) if score > 0 else 0.0
            hits.append(SparseHit(chunk_id=chunk_id, score=normalized, topic_id=topic_id))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


class PyseriniSparseBackend:
    """Lucene BM25 backend powered by pyserini."""

    def __init__(self, chunks: Sequence, chunk_topic_map: Dict[int, int]):
        self.logger = Logger().get_logger("SparsePyserini")
        self.chunk_topic_map = chunk_topic_map
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.index_dir: Optional[Path] = None
        self.searcher = None

        self._build_index(chunks)

    def _build_index(self, chunks: Sequence):
        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:
            raise RuntimeError(
                "pyserini is not available. Install with `pip install pyserini`."
            ) from exc

        self._temp_dir = tempfile.TemporaryDirectory(prefix="pyserini_sparse_")
        temp_root = Path(self._temp_dir.name)
        corpus_dir = temp_root / "corpus"
        index_dir = temp_root / "index"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = corpus_dir / "docs.json"
        with corpus_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                payload = {"id": str(chunk.chunk_id), "contents": chunk.text}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        cmd = [
            "python",
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(corpus_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(max(1, min(4, os.cpu_count() or 1))),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to build pyserini index. "
                f"stdout={proc.stdout[-600:]} stderr={proc.stderr[-600:]}"
            )

        self.index_dir = index_dir
        self.searcher = LuceneSearcher(str(index_dir))

    def search(self, query: str, topic_ids: Sequence[int], k: int) -> List[SparseHit]:
        if self.searcher is None:
            return []

        allowed_topics = set(topic_ids)
        # Fetch a wider set before topic filtering.
        raw_hits = self.searcher.search(query, k=max(50, k * 5))

        hits: List[SparseHit] = []
        seen = set()

        for hit in raw_hits:
            try:
                chunk_id = int(hit.docid)
            except ValueError:
                continue

            if chunk_id in seen:
                continue

            topic_id = self.chunk_topic_map.get(chunk_id, -1)
            if topic_id not in allowed_topics:
                continue

            score = float(hit.score)
            normalized = float(score / (score + 1.0)) if score > 0 else 0.0
            hits.append(SparseHit(chunk_id=chunk_id, score=normalized, topic_id=topic_id))
            seen.add(chunk_id)

            if len(hits) >= k:
                break

        return hits

    def close(self):
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        if self.index_dir is not None and self.index_dir.exists():
            shutil.rmtree(self.index_dir, ignore_errors=True)
            self.index_dir = None


class SparseRetriever:
    """Configurable sparse retriever with pyserini support."""

    def __init__(self, chunks: Sequence, topic_chunk_maps: Dict[int, List[int]]):
        self.logger = Logger().get_logger("SparseRetriever")
        self.chunk_topic_map: Dict[int, int] = {}
        for topic_id, chunk_ids in topic_chunk_maps.items():
            for chunk_id in chunk_ids:
                self.chunk_topic_map[int(chunk_id)] = int(topic_id)

        preferred_backend = os.getenv("SPARSE_RETRIEVER_BACKEND", "rank_bm25").lower()
        self.backend_name = "rank_bm25"

        if preferred_backend == "pyserini":
            try:
                self.backend = PyseriniSparseBackend(chunks, self.chunk_topic_map)
                self.backend_name = "pyserini"
                self.logger.info("Sparse retriever initialized with pyserini backend")
            except Exception as exc:
                self.logger.warning(
                    f"pyserini backend unavailable ({exc}); falling back to rank_bm25"
                )
                self.backend = RankBM25SparseBackend(chunks, self.chunk_topic_map)
        else:
            self.backend = RankBM25SparseBackend(chunks, self.chunk_topic_map)

    def search(self, query: str, topic_ids: Sequence[int], k: int) -> List[Tuple[int, float, int]]:
        hits = self.backend.search(query=query, topic_ids=topic_ids, k=k)
        return [(hit.chunk_id, hit.score, hit.topic_id) for hit in hits]

    def close(self):
        close_fn = getattr(self.backend, "close", None)
        if callable(close_fn):
            close_fn()

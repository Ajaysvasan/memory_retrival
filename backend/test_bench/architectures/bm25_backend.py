"""BM25 backend abstraction with optional Pyserini support."""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from bench_core.document import Document


class BM25Backend:
    """Provides BM25 scores using Pyserini (default) or rank-bm25 fallback."""

    SUPPORTED_BACKENDS = {"rank_bm25", "pyserini"}

    def __init__(self):
        requested = os.getenv("TEST_BENCH_BM25_BACKEND", "pyserini").strip().lower()
        if requested not in self.SUPPORTED_BACKENDS:
            requested = "rank_bm25"

        self.requested_backend = requested
        self.active_backend = "rank_bm25"

        self.logger = logging.getLogger(__name__)
        self.documents: List[Document] = []
        self.doc_id_to_index: Dict[str, int] = {}

        self.rank_bm25_index = None
        self.pyserini_searcher = None

    def train(self, documents: List[Document]):
        """Build index for the selected backend; fallback to rank-bm25 when needed."""
        self.documents = documents
        self.doc_id_to_index = {str(doc.doc_id): idx for idx, doc in enumerate(documents)}

        if self.requested_backend == "pyserini":
            try:
                self._build_pyserini_index(documents)
                self.active_backend = "pyserini"
                return
            except Exception as exc:  # pragma: no cover - fallback path
                self.logger.warning(
                    "Pyserini backend unavailable (%s). Falling back to rank-bm25.",
                    exc,
                )

        self._build_rank_bm25_index(documents)
        self.active_backend = "rank_bm25"

    def score_all(self, query: str) -> np.ndarray:
        """Return BM25 scores aligned with self.documents order."""
        if self.active_backend == "pyserini" and self.pyserini_searcher is not None:
            hits = self.pyserini_searcher.search(query, k=len(self.documents))
            score_map = {str(hit.docid): float(hit.score) for hit in hits}
            return np.array(
                [score_map.get(str(doc.doc_id), 0.0) for doc in self.documents],
                dtype=float,
            )

        if self.rank_bm25_index is None:
            raise RuntimeError("BM25 backend not trained")

        query_tokens = query.lower().split()
        return np.asarray(self.rank_bm25_index.get_scores(query_tokens), dtype=float)

    def top_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """Return top-k documents for the query."""
        if self.active_backend == "pyserini" and self.pyserini_searcher is not None:
            hits = self.pyserini_searcher.search(query, k=top_k)
            selected_docs = []
            for hit in hits:
                idx = self.doc_id_to_index.get(str(hit.docid))
                if idx is not None:
                    selected_docs.append(self.documents[idx])
            return selected_docs

        scores = self.score_all(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]

    def _build_rank_bm25_index(self, documents: List[Document]):
        tokenized_corpus = [doc.content.lower().split() for doc in documents]
        self.rank_bm25_index = BM25Okapi(tokenized_corpus)

    def _build_pyserini_index(self, documents: List[Document]):
        from pyserini.search.lucene import LuceneSearcher

        root_dir = Path(
            os.getenv("TEST_BENCH_PYSERINI_INDEX_DIR", ".pyserini_index")
        ).resolve()
        docs_dir = root_dir / "docs"
        index_dir = root_dir / "index"

        if root_dir.exists():
            shutil.rmtree(root_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)

        docs_path = docs_dir / "docs.jsonl"
        with docs_path.open("w", encoding="utf-8") as f:
            for doc in documents:
                payload = {"id": str(doc.doc_id), "contents": doc.content}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(docs_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())

        self.pyserini_searcher = LuceneSearcher(str(index_dir))

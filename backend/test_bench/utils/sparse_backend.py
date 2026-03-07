"""Sparse retrieval backends for the test bench.

Backend selection:
- rank_bm25 (default)
- pyserini (set SPARSE_RETRIEVER_BACKEND=pyserini)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from bench_core.document import Document


class SparseRetrievalBackend:
    """Sparse retrieval wrapper with optional pyserini backend."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]
        self.backend = os.getenv("SPARSE_RETRIEVER_BACKEND", "rank_bm25").lower()

        self._bm25 = None
        self._pyserini_searcher = None
        self._tmpdir = None

        if self.backend == "pyserini":
            try:
                self._init_pyserini()
            except Exception as exc:
                print(
                    f"[SparseRetrievalBackend] pyserini unavailable ({exc}), falling back to rank_bm25"
                )
                self.backend = "rank_bm25"
                self._init_rank_bm25()
        else:
            self._init_rank_bm25()

    def _init_rank_bm25(self):
        tokenized_corpus = [doc.content.lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _init_pyserini(self):
        from pyserini.search.lucene import LuceneSearcher

        self._tmpdir = tempfile.TemporaryDirectory(prefix="test_bench_pyserini_")
        root = Path(self._tmpdir.name)
        corpus_dir = root / "corpus"
        index_dir = root / "index"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = corpus_dir / "docs.json"
        with corpus_path.open("w", encoding="utf-8") as handle:
            for doc in self.documents:
                payload = {"id": str(doc.doc_id), "contents": doc.content}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
                "failed to build pyserini index: "
                f"stdout={proc.stdout[-500:]} stderr={proc.stderr[-500:]}"
            )

        self._pyserini_searcher = LuceneSearcher(str(index_dir))

    def get_scores(self, query: str) -> np.ndarray:
        """Return sparse scores aligned with self.documents order."""

        if self.backend == "pyserini" and self._pyserini_searcher is not None:
            scores = np.zeros(len(self.documents), dtype=float)
            hits = self._pyserini_searcher.search(query, k=max(1, len(self.documents)))
            score_map = {str(hit.docid): float(hit.score) for hit in hits}

            for idx, doc_id in enumerate(self.doc_ids):
                raw_score = score_map.get(str(doc_id), 0.0)
                scores[idx] = raw_score

            return scores

        # rank_bm25 fallback/default
        query_tokens = query.lower().split()
        return np.array(self._bm25.get_scores(query_tokens), dtype=float)

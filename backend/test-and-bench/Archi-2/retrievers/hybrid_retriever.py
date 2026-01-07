from vectordb.vector_db import VectorDatabase
from retrievers.bm25_index import BM25Index

class HybridRetriever:
    def __init__(self, alpha=0.6):
        self.vector_db = VectorDatabase()
        self.bm25 = BM25Index()
        self.alpha = alpha

    def index(self, documents):
        self.docs = documents
        self.vector_db.add_documents(documents)
        self.bm25.index(documents)

    def search(self, query, top_k=10):
        v = self.vector_db.search(query, top_k * 2)
        b = self.bm25.search(query, top_k * 2)

        scores = {}
        for d in v:
            scores[d.doc_id] = self.alpha * d.retrieval_score
        for d in b:
            scores[d.doc_id] = scores.get(d.doc_id, 0) + (1 - self.alpha) * d.retrieval_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        lookup = {d.doc_id: d for d in v + b}
        return [lookup[i] for i, _ in ranked]

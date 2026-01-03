from .semantic import SemanticRetriever
from .bm25 import BM25Retriever
from core.retrieved_doc import RetrievedDoc

class HybridRetriever:

    def __init__(self, alpha=0.5):
        self.semantic = SemanticRetriever()
        self.bm25 = BM25Retriever()
        self.alpha = alpha
        self.documents = []

    def index(self, documents):
        self.documents = documents
        self.semantic.index(documents)
        self.bm25.index(documents)

    def retrieve(self, query, top_k=10):
        sem = dict(self.semantic.retrieve(query, top_k * 2))
        bm = dict(self.bm25.retrieve(query, top_k * 2))

        all_ids = set(sem) | set(bm)
        results = []

        for doc_id in all_ids:
            score = self.alpha * sem.get(doc_id, 0) + (1 - self.alpha) * bm.get(doc_id, 0)
            doc = next(d for d in self.documents if d.doc_id == doc_id)
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    content=doc.content,
                    semantic_score=sem.get(doc_id, 0),
                    bm25_score=bm.get(doc_id, 0),
                    hybrid_score=score,
                    metadata=doc.metadata
                )
            )

        return sorted(results, key=lambda x: x.hybrid_score, reverse=True)[:top_k]

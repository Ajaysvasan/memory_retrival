from core.retrieved_doc import RetrievedDoc
from .semantic import SemanticRetriever
from .bm25 import BM25RetrieverModule


class HybridRetriever:

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.semantic = SemanticRetriever()
        self.bm25 = BM25RetrieverModule()
        self.documents = []

    def index(self, documents):
        self.documents = documents
        self.semantic.index(documents)
        self.bm25.index(documents)

    def retrieve(self, query, top_k=10):
        sem = dict(self.semantic.retrieve(query, top_k * 2))
        bm = dict(self.bm25.retrieve(query, top_k * 2))

        results = []
        for doc in self.documents:
            score = self.alpha * sem.get(doc.doc_id, 0) + (1 - self.alpha) * bm.get(doc.doc_id, 0)
            results.append(
                RetrievedDoc(doc.doc_id, doc.content, score, doc.metadata)
            )

        return sorted(results, key=lambda x: x.retrieval_score, reverse=True)[:top_k]

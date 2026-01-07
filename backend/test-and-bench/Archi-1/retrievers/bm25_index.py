import numpy as np
from rank_bm25 import BM25Okapi
from core.retrieved_doc import RetrievedDoc

class BM25Index:
    def index(self, documents):
        self.docs = documents
        self.bm25 = BM25Okapi([d.content.lower().split() for d in documents])

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDoc(self.docs[i].doc_id, self.docs[i].content,
                         bm25_score=float(scores[i]), metadata=self.docs[i].metadata)
            for i in idx
        ]

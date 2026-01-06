import numpy as np
from rank_bm25 import BM25Okapi
from tools.base import BaseTool
from core.retrieved_doc import RetrievedDoc

class BM25SearchTool(BaseTool):
    def __init__(self, documents):
        super().__init__("bm25_search")
        self.docs = documents
        self.bm25 = BM25Okapi([d.content.lower().split() for d in documents])

    def execute(self, query, top_k=5):
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedDoc(self.docs[i].doc_id, self.docs[i].content,
                         float(scores[i]), self.docs[i].metadata)
            for i in idx
        ]

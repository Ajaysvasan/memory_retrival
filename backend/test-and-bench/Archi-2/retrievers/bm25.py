import numpy as np
from rank_bm25 import BM25Okapi
from .base import RetrieverModule


class BM25RetrieverModule(RetrieverModule):

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def index(self, documents):
        self.documents = documents
        corpus = [d.content.lower().split() for d in documents]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in idx]

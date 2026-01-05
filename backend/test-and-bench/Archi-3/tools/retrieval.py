import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from core.retrieved_doc import RetrievedDoc

class HybridSearchTool:
    def __init__(self, documents):
        self.docs = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embs = self.model.encode([d.content for d in documents])
        self.bm25 = BM25Okapi([d.content.lower().split() for d in documents])

    def execute(self, query, top_k=5):
        q = self.model.encode(query)
        sem = self.embs @ q / (np.linalg.norm(self.embs, axis=1)*np.linalg.norm(q)+1e-8)
        bm = self.bm25.get_scores(query.lower().split())
        scores = 0.6 * sem + 0.4 * bm
        idx = scores.argsort()[::-1][:top_k]

        return [
            RetrievedDoc(self.docs[i].doc_id, self.docs[i].content, float(scores[i]), self.docs[i].metadata)
            for i in idx
        ]

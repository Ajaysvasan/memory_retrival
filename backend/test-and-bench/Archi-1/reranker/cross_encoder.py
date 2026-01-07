import numpy as np
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    def rerank(self, query, docs, top_k):
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]

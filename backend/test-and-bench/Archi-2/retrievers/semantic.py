import numpy as np
from sentence_transformers import SentenceTransformer
from .base import RetrieverModule


class SemanticRetriever(RetrieverModule):

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = None

    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(
            [d.content for d in documents],
            convert_to_numpy=True
        )

    def retrieve(self, query, top_k=10):
        q = self.model.encode(query, convert_to_numpy=True)
        scores = self.embeddings @ q / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q) + 1e-8
        )
        idx = scores.argsort()[::-1][:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in idx]

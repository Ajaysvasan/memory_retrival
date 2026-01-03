import numpy as np
from sentence_transformers import SentenceTransformer
from .base import RetrieverModule

class SemanticRetriever(RetrieverModule):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(
            [d.content for d in documents],
            convert_to_tensor=True
        )

    def retrieve(self, query, top_k=10):
        q = self.model.encode(query, convert_to_tensor=True)
        scores = np.array([
            float(q.dot(e)) /
            (np.linalg.norm(q) * np.linalg.norm(e) + 1e-8)
            for e in self.embeddings
        ])
        idx = scores.argsort()[::-1][:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in idx]

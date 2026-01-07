import numpy as np
from sentence_transformers import SentenceTransformer
from core.vector_record import VectorRecord
from core.retrieved_doc import RetrievedDoc

class VectorDatabase:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.vectors = {}
        self.embedding_dim = 384

    def add_documents(self, documents):
        embeddings = self.model.encode([d.content for d in documents])
        for d, e in zip(documents, embeddings):
            self.vectors[d.doc_id] = VectorRecord(
                d.doc_id, np.array(e, dtype=np.float32), d.content, d.metadata
            )

    def search(self, query, top_k=10):
        q = self.model.encode(query)
        sims = {
            k: float(np.dot(q, v.embedding) /
                     (np.linalg.norm(q) * np.linalg.norm(v.embedding) + 1e-8))
            for k, v in self.vectors.items()
        }
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedDoc(i, self.vectors[i].content, semantic_score=s, metadata=self.vectors[i].metadata)
            for i, s in ranked
        ]

    def get_stats(self):
        return {
            "total_documents": len(self.vectors),
            "embedding_dimension": self.embedding_dim,
            "memory_usage_mb": (len(self.vectors)*self.embedding_dim*4)/(1024*1024)
        }

import numpy as np
from sentence_transformers import SentenceTransformer
from core.vector_embedding import VectorEmbedding
from core.retrieved_doc import RetrievedDoc

class InMemoryVectorDB:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectors = {}

    def add_documents(self, docs):
        embeddings = self.model.encode([d.content for d in docs])
        for d, e in zip(docs, embeddings):
            self.vectors[d.doc_id] = VectorEmbedding(
                d.doc_id, np.array(e, dtype=np.float32), d.content, d.metadata
            )

    def search(self, query, top_k=5):
        q = self.model.encode(query)
        scores = {}
        for k, v in self.vectors.items():
            scores[k] = float(np.dot(q, v.embedding) /
                (np.linalg.norm(q)*np.linalg.norm(v.embedding)+1e-8))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedDoc(doc_id=i, content=self.vectors[i].content,
                         retrieval_score=s, metadata=self.vectors[i].metadata)
            for i, s in ranked
        ]
import torch
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from core.retrieved_doc import RetrievedDoc
from .fusion_encoder import FusionEncoder


class AnswerGeneratorFiD:
    """
    Answer generator using FiD-style fusion + extractive fallback.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[RetrievedDoc],
        fusion_encoder: Optional[FusionEncoder] = None
    ) -> Tuple[str, Dict[str, Any]]:

        if not retrieved_docs:
            return "No relevant documents found.", {"fusion_applied": False}

        query_emb = torch.tensor(self.embedding_model.encode(query)).float()
        doc_embs = [
            torch.tensor(self.embedding_model.encode(doc.content)).float()
            for doc in retrieved_docs
        ]

        fusion_stats = {
            "num_documents_fused": len(doc_embs),
            "query_embedding_dim": query_emb.shape[0],
            "fusion_applied": False,
        }

        if fusion_encoder is not None:
            with torch.no_grad():
                _, attn = fusion_encoder(query_emb, doc_embs)
                fusion_stats["attention_weights"] = attn.tolist()
                fusion_stats["fusion_applied"] = True

        answer = self._extractive_generation(query, retrieved_docs)
        return answer, fusion_stats

    @staticmethod
    def _extractive_generation(query: str, docs: List[RetrievedDoc]) -> str:
        combined = " ".join(doc.content for doc in docs[:3])
        sentences = combined.split(".")
        q_tokens = set(query.lower().split())

        selected = []
        for s in sentences:
            if len(q_tokens & set(s.lower().split())) > 1:
                selected.append(s.strip())

        if selected:
            return ". ".join(selected[:4]) + "."

        return combined[:250] + "..."

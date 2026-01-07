import torch
from sentence_transformers import SentenceTransformer

class AnswerGenerator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate(self, query, docs, fusion_encoder=None):
        q = torch.tensor(self.model.encode(query)).float()
        d = [torch.tensor(self.model.encode(x.content)).float() for x in docs]

        stats = {"fusion_applied": fusion_encoder is not None}
        if fusion_encoder:
            _, w = fusion_encoder(q, d)
            stats["attention_weights"] = w.tolist()

        text = " ".join(x.content for x in docs[:3])
        return text[:300] + "...", stats

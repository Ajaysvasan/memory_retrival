import torch
import torch.nn as nn


class FusionEncoder(nn.Module):

    def __init__(self, embedding_dim=384, hidden_dim=768):
        super().__init__()
        self.project = nn.Linear(embedding_dim * 2, embedding_dim)
        self.attn = nn.Linear(embedding_dim, 1)
        self.fusion = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, query_emb, doc_embs):
        contexts = []
        for d in doc_embs:
            ctx = self.project(torch.cat([query_emb, d], dim=0))
            contexts.append(ctx)

        stacked = torch.stack(contexts)
        weights = torch.softmax(self.attn(stacked), dim=0)
        fused = (stacked * weights).sum(dim=0)
        return self.fusion(fused), weights.squeeze()

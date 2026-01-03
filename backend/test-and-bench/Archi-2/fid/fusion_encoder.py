import torch
import torch.nn as nn
from typing import List, Tuple


class FusionEncoder(nn.Module):
    """
    Fusion-in-Decoder Encoder:
    Encodes each [query, document] pair independently and fuses them with attention.
    """

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 768, num_documents: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_documents = num_documents

        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )

        self.attention_weights = nn.Linear(embedding_dim, 1)

        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * num_documents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        doc_embs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        context_encodings = []

        for doc_emb in doc_embs[:self.num_documents]:
            context = torch.cat([query_emb, doc_emb], dim=-1).unsqueeze(1)

            if context.shape[-1] > self.embedding_dim:
                context = context[..., :self.embedding_dim]
            else:
                padding = self.embedding_dim - context.shape[-1]
                context = torch.nn.functional.pad(context, (0, padding))

            encoded = self.context_encoder(context)
            context_encodings.append(encoded.squeeze(1))

        stacked = torch.stack(context_encodings)  # [k, 1, d]

        attn = self.attention_weights(stacked)
        attn = torch.softmax(attn, dim=0)

        weighted = stacked * attn
        fused = self.fusion_layer(weighted.flatten())

        return fused.unsqueeze(0), attn.squeeze(-1).squeeze(-1)

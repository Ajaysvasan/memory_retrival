import torch
import torch.nn as nn


class FusionDecoder(nn.Module):
    """
    Transformer-based decoder for FiD fused representations.
    """

    def __init__(self, hidden_dim: int = 768, vocab_size: int = 30000, max_length: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=2
        )

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, fused_repr: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(tgt, fused_repr.unsqueeze(0))
        return self.output_projection(decoded)

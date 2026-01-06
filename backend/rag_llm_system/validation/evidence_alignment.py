from typing import List

import numpy as np

from config.thresholds import THRESHOLDS, Chunk
from ingest.embedder import Embedder


class EvidenceAligner:
    """Check claim-evidence alignment."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def compute_alignment(self, claims: List[str], chunks: List[Chunk]) -> float:
        """Compute evidence alignment score."""
        if not claims or not chunks:
            return 0.0

        supported = 0
        chunk_texts = [c.text for c in chunks]

        for claim in claims:
            if self._is_supported(claim, chunk_texts):
                supported += 1

        return supported / len(claims)

    def _is_supported(self, claim: str, evidence_texts: List[str]) -> bool:
        """Check if claim is supported by evidence."""
        claim_emb = self.embedder.embed_query(claim)
        evidence_embs = self.embedder.embed_batch(evidence_texts)

        # Max similarity with any evidence
        similarities = np.dot(evidence_embs, claim_emb)
        max_sim = np.max(similarities) if len(similarities) > 0 else 0

        return max_sim > THRESHOLDS.NLI_ENTAILMENT_THRESHOLD

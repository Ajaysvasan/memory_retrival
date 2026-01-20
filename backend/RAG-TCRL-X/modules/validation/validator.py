import re
from typing import List, Set, Tuple, FrozenSet
import numpy as np
from core.contracts.retrieve_result import RetrieveResult
from core.contracts.validation import Validation, ValidationStatus
from logger import Logger
from config import Config


class Validator:
    """Validation engine with claim extraction and evidence alignment"""

    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.logger = Logger().get_logger("Validator")

    def validate(self, answer: str, retrieval_result: RetrieveResult) -> Validation:
        """Validate answer against retrieved evidence"""

        # Extract claims from answer
        claims = self._extract_claims(answer)

        if not claims:
            return Validation(
                status=ValidationStatus.NO_CLAIMS,
                evidence_score=0.0,
                claims=(),
                evidence_chunk_ids=frozenset(),
                reasoning="No verifiable claims in answer",
            )

        # Check for over-information
        if len(claims) > 10 or len(answer.split()) > 500:
            return Validation(
                status=ValidationStatus.OVER_INFORMATION,
                evidence_score=0.0,
                claims=tuple(claims),
                evidence_chunk_ids=frozenset(),
                reasoning="Answer contains too much information",
            )

        # Compute evidence alignment
        evidence_score, aligned_chunks = self._compute_evidence_alignment(
            claims, retrieval_result
        )

        # Check for contradictions
        contradictions = self._detect_contradictions(claims, retrieval_result)

        # Determine validation status
        if contradictions:
            status = ValidationStatus.CONTRADICTION_DETECTED
        elif evidence_score < Config.EVIDENCE_THRESHOLD:
            status = ValidationStatus.INSUFFICIENT_EVIDENCE
        else:
            status = ValidationStatus.VALID

        validation = Validation(
            status=status,
            evidence_score=evidence_score,
            claims=tuple(claims),
            evidence_chunk_ids=frozenset(aligned_chunks),
            contradictions=tuple(contradictions),
            reasoning=f"Evidence score: {evidence_score:.2f}, Claims: {len(claims)}",
        )

        self.logger.debug(f"Validation: {status.value} (score={evidence_score:.2f})")
        return validation

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text"""
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Filter out questions and meta-statements
            if sentence.endswith("?"):
                continue
            if any(
                phrase in sentence.lower()
                for phrase in ["i think", "i believe", "in my opinion", "it seems"]
            ):
                continue

            claims.append(sentence)

        return claims

    def _compute_evidence_alignment(
        self, claims: List[str], retrieval_result: RetrieveResult
    ) -> Tuple[float, Set[int]]:
        """Compute evidence alignment score using semantic similarity"""

        if not claims or not retrieval_result.chunks:
            return 0.0, set()

        # Get embeddings for claims
        claim_embeddings = self.embedding_engine.embed_texts(claims)

        # Get embeddings for evidence chunks
        evidence_texts = [chunk.text for chunk in retrieval_result.chunks]
        evidence_embeddings = self.embedding_engine.embed_texts(evidence_texts)

        # Compute similarity matrix
        similarities = np.dot(claim_embeddings, evidence_embeddings.T)

        # For each claim, find max similarity with evidence
        aligned_chunks = set()
        supported_claims = 0

        for i, claim_sims in enumerate(similarities):
            max_sim = np.max(claim_sims)
            if max_sim >= Config.SIMILARITY_THRESHOLD:
                supported_claims += 1
                # Track which chunk provided evidence
                best_chunk_idx = np.argmax(claim_sims)
                aligned_chunks.add(retrieval_result.chunks[best_chunk_idx].chunk_id)

        # Evidence score = ratio of supported claims
        evidence_score = supported_claims / len(claims) if claims else 0.0

        return evidence_score, aligned_chunks

    def _detect_contradictions(
        self, claims: List[str], retrieval_result: RetrieveResult
    ) -> List[str]:
        """Detect contradictions between claims and evidence"""

        if not claims or not retrieval_result.chunks:
            return []

        contradictions = []

        # Get embeddings
        claim_embeddings = self.embedding_engine.embed_texts(claims)
        evidence_texts = [chunk.text for chunk in retrieval_result.chunks]
        evidence_embeddings = self.embedding_engine.embed_texts(evidence_texts)

        # Look for negation patterns
        negation_patterns = [
            (r"\bnot\b", r"\bis\b"),
            (r"\bno\b", r"\byes\b"),
            (r"\bnever\b", r"\balways\b"),
            (r"\bfalse\b", r"\btrue\b"),
        ]

        for i, claim in enumerate(claims):
            claim_lower = claim.lower()

            for evidence_text in evidence_texts:
                evidence_lower = evidence_text.lower()

                # Check for explicit negation patterns
                for neg_pattern, pos_pattern in negation_patterns:
                    if re.search(neg_pattern, claim_lower) and re.search(
                        pos_pattern, evidence_lower
                    ):

                        # Verify with semantic similarity
                        sim = np.dot(
                            claim_embeddings[i],
                            evidence_embeddings[evidence_texts.index(evidence_text)],
                        )

                        if sim > Config.CONTRADICTION_THRESHOLD:
                            contradictions.append(
                                f"Claim '{claim}' contradicts evidence"
                            )
                            break

        return contradictions

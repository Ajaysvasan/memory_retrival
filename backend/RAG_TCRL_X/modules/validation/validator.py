import re
from typing import FrozenSet, List, Set, Tuple

import numpy as np

from config import Config
from core.contracts.query import Query
from core.contracts.retrieve_result import RetrieveResult
from core.contracts.validation import Validation, ValidationStatus
from logger import Logger


class Validator:
    """Validation engine with claim extraction and evidence alignment"""

    def __init__(self, embedding_engine):
        self.embedding_engine = embedding_engine
        self.logger = Logger().get_logger("Validator")

        self.query_evidence_threshold = 0.30
        self.query_answer_threshold = 0.35

    def validate(
        self, answer: str, retrieval_result: RetrieveResult, query: Query = None
    ) -> Validation:
        """Validate answer against retrieved evidence and query"""

        claims = self._extract_claims(answer)

        if not claims:
            return Validation(
                status=ValidationStatus.NO_CLAIMS,
                evidence_score=0.0,
                claims=(),
                evidence_chunk_ids=frozenset(),
                reasoning="No verifiable claims in answer",
            )

        if len(claims) > 10 or len(answer.split()) > 500:
            return Validation(
                status=ValidationStatus.OVER_INFORMATION,
                evidence_score=0.0,
                claims=tuple(claims),
                evidence_chunk_ids=frozenset(),
                reasoning="Answer contains too much information",
            )

        evidence_texts = [chunk.text for chunk in retrieval_result.chunks]

        if not evidence_texts:
            return Validation(
                status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                evidence_score=0.0,
                claims=tuple(claims),
                evidence_chunk_ids=frozenset(),
                reasoning="No evidence chunks available",
            )

        evidence_embeddings = self.embedding_engine.embed_texts(evidence_texts)

        if query is not None:
            query_embedding = self.embedding_engine.embed_query(query.text)

            query_evidence_score = self._query_evidence_alignment(
                query_embedding, evidence_embeddings
            )

            self.logger.debug(f"Query-Evidence alignment: {query_evidence_score:.3f}")

            if query_evidence_score < self.query_evidence_threshold:
                return Validation(
                    status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                    evidence_score=0.0,
                    claims=tuple(claims),
                    evidence_chunk_ids=frozenset(),
                    reasoning="No retrieved evidence is relevant to the query.",
                )

            answer_embedding = self.embedding_engine.embed_query(answer)

            query_answer_score = self._query_answer_alignment(
                query_embedding, answer_embedding
            )

            self.logger.debug(f"Query-Answer alignment: {query_answer_score:.3f}")

            if query_answer_score < self.query_answer_threshold:
                return Validation(
                    status=ValidationStatus.INSUFFICIENT_EVIDENCE,
                    evidence_score=0.0,
                    claims=tuple(claims),
                    evidence_chunk_ids=frozenset(),
                    reasoning="Generated answer does not address the question.",
                )

        evidence_score, aligned_chunks = self._compute_evidence_alignment(
            claims, retrieval_result, evidence_embeddings
        )

        contradictions = self._detect_contradictions(
            claims, retrieval_result, evidence_embeddings
        )

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

    def _query_evidence_alignment(
        self, query_embedding: np.ndarray, evidence_embeddings: np.ndarray
    ) -> float:
        """Compute query-evidence alignment using max similarity"""

        if len(evidence_embeddings) == 0:
            return 0.0

        similarities = np.dot(evidence_embeddings, query_embedding)

        max_similarity = np.max(similarities)

        return float(max_similarity)

    def _query_answer_alignment(
        self, query_embedding: np.ndarray, answer_embedding: np.ndarray
    ) -> float:
        """Compute query-answer alignment using cosine similarity"""

        similarity = np.dot(query_embedding, answer_embedding)

        return float(similarity)

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text"""

        sentences = re.split(r"[.!?]+", text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

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
        self,
        claims: List[str],
        retrieval_result: RetrieveResult,
        evidence_embeddings: np.ndarray = None,
    ) -> Tuple[float, Set[int]]:
        """Compute evidence alignment score using semantic similarity"""

        if not claims or not retrieval_result.chunks:
            return 0.0, set()

        claim_embeddings = self.embedding_engine.embed_texts(claims)

        if evidence_embeddings is None:
            evidence_texts = [chunk.text for chunk in retrieval_result.chunks]
            evidence_embeddings = self.embedding_engine.embed_texts(evidence_texts)

        similarities = np.dot(claim_embeddings, evidence_embeddings.T)

        aligned_chunks = set()
        supported_claims = 0

        for i, claim_sims in enumerate(similarities):
            max_sim = np.max(claim_sims)
            if max_sim >= Config.SIMILARITY_THRESHOLD:
                supported_claims += 1
                best_chunk_idx = np.argmax(claim_sims)
                aligned_chunks.add(retrieval_result.chunks[best_chunk_idx].chunk_id)

        evidence_score = supported_claims / len(claims) if claims else 0.0

        return evidence_score, aligned_chunks

    def _detect_contradictions(
        self,
        claims: List[str],
        retrieval_result: RetrieveResult,
        evidence_embeddings: np.ndarray = None,
    ) -> List[str]:
        """Detect contradictions between claims and evidence"""

        if not claims or not retrieval_result.chunks:
            return []

        contradictions = []

        claim_embeddings = self.embedding_engine.embed_texts(claims)

        if evidence_embeddings is None:
            evidence_texts = [chunk.text for chunk in retrieval_result.chunks]
            evidence_embeddings = self.embedding_engine.embed_texts(evidence_texts)

        negation_patterns = [
            (r"\bnot\b", r"\bis\b"),
            (r"\bno\b", r"\byes\b"),
            (r"\bnever\b", r"\balways\b"),
            (r"\bfalse\b", r"\btrue\b"),
        ]

        evidence_texts = [chunk.text for chunk in retrieval_result.chunks]

        for i, claim in enumerate(claims):
            claim_lower = claim.lower()

            for j, evidence_text in enumerate(evidence_texts):
                evidence_lower = evidence_text.lower()

                for neg_pattern, pos_pattern in negation_patterns:
                    if re.search(neg_pattern, claim_lower) and re.search(
                        pos_pattern, evidence_lower
                    ):

                        sim = np.dot(claim_embeddings[i], evidence_embeddings[j])

                        if sim > Config.CONTRADICTION_THRESHOLD:
                            contradictions.append(
                                f"Claim '{claim}' contradicts evidence"
                            )
                            break

        return contradictions

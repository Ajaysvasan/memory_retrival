from typing import Optional
from core.contracts.retrieve_result import RetrieveResult
from core.contracts.query import Query
from logger import Logger


class GeneratorAdaptor:
    """Adapter for answer generation (placeholder for LLM integration)"""

    def __init__(self):
        self.logger = Logger().get_logger("Generator")

    def generate(self, query: Query, retrieval_result: RetrieveResult) -> str:
        """Generate answer from query and retrieved chunks"""

        if not retrieval_result.chunks:
            return "I don't have enough information to answer this question."

        # Simple extractive generation (placeholder)
        # In production, this would use an LLM

        context_parts = []
        for chunk in retrieval_result.chunks[:3]:  # Top 3 chunks
            context_parts.append(chunk.text)

        context = "\n\n".join(context_parts)

        # Create simple answer
        answer = f"Based on the available information:\n\n{context}"

        self.logger.debug(
            f"Generated answer with {len(retrieval_result.chunks)} chunks"
        )
        return answer

    def generate_refusal(self, reason: str) -> str:
        """Generate refusal message"""
        return f"I cannot answer this question. {reason}"

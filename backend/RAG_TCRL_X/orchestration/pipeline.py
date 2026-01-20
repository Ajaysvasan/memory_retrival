import time
from datetime import datetime
from typing import Dict, Optional

from config import Config
from core.contracts.query import Query
from core.contracts.retrieve_result import RetrieveResult
from core.contracts.validation import Validation
from core.errors.refusal_reason import RefusalReason
from logger import Logger
from modules.generation.generator_adaptor import GeneratorAdaptor
from modules.intake.query_intake import QueryIntake
from modules.memory_gate.mutation_gate import MutationGate
from modules.rl.rl_agent import RLAgent
from modules.validation.validator import Validator
from orchestration.phase_a_orchestrator import PhaseAOrchestrator


class Pipeline:
    """Main RAG-TCRL-X pipeline orchestrator"""

    def __init__(
        self,
        phase_a_orchestrator: PhaseAOrchestrator,
        retrieval_engine,
        validator: Validator,
        generator: GeneratorAdaptor,
        mutation_gate: MutationGate,
        rl_agent: RLAgent,
    ):

        self.query_intake = QueryIntake()
        self.phase_a = phase_a_orchestrator
        self.retrieval_engine = retrieval_engine
        self.validator = validator
        self.generator = generator
        self.mutation_gate = mutation_gate
        self.rl_agent = rl_agent

        self.logger = Logger().get_logger("Pipeline")

        self.query_count = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.refusals = 0

    def process(self, query_text: str, user_id: Optional[str] = None) -> Dict:
        """Process query through full pipeline"""

        start_time = time.time()
        self.query_count += 1

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing Query #{self.query_count}: {query_text[:100]}...")
        self.logger.info(f"{'='*80}")

        try:
            query = self.query_intake.process(query_text, user_id=user_id)
            self.logger.info(f"[1] Query intake complete")

            context = self._build_context()

            phase_a_decision = self.phase_a.orchestrate(query, context)
            self.logger.info(
                f"[2] Phase A complete: should_proceed={phase_a_decision.should_proceed}"
            )

            if not phase_a_decision.should_proceed:
                self.refusals += 1
                return self._create_refusal_response(
                    phase_a_decision.refusal_reason, start_time
                )

            cached_chunks = None
            if phase_a_decision.plan.use_cache:
                cached_chunks = self.mutation_gate.check_cache(phase_a_decision.plan)
                if cached_chunks:
                    self.logger.info(f"[3] Cache HIT: {len(cached_chunks)} chunk IDs")

            if cached_chunks:
                self.cache_hits += 1
                retrieval_result = self._create_cached_result(cached_chunks)
            else:
                self.logger.info(f"[3] Cache MISS - invoking retrieval engine")
                retrieval_result = self.retrieval_engine.retrieve(
                    query, phase_a_decision.plan
                )
                self.logger.info(
                    f"[4] Retrieval complete: {len(retrieval_result.chunks)} chunks retrieved"
                )

            if len(retrieval_result.chunks) == 0:
                self.logger.error("RETRIEVAL FAILURE: 0 chunks returned")
                self.refusals += 1
                return self._create_refusal_response(
                    RefusalReason.NO_RELEVANT_DATA.to_message(), start_time
                )

            answer = self.generator.generate(query, retrieval_result)
            self.logger.info(f"[5] Generation complete: {len(answer)} chars")

            validation = self.validator.validate(answer, retrieval_result, query)
            self.logger.info(
                f"[6] Validation complete: status={validation.status.value}, score={validation.evidence_score:.2f}"
            )

            if validation.should_refuse:
                self.refusals += 1
                return self._create_validation_refusal(validation, start_time)

            self._update_memory_gate(
                phase_a_decision.plan, retrieval_result, validation
            )

            latency_ms = (time.time() - start_time) * 1000
            self._train_rl(context, validation, latency_ms)

            return self._create_success_response(
                answer, validation, retrieval_result, latency_ms
            )

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._create_error_response(str(e), start_time)

    def _build_context(self) -> Dict:
        """Build context for RL agent"""
        cache_hit_rate = self.cache_hits / max(1, self.query_count)
        avg_latency = self.total_latency / max(1, self.query_count)

        return {
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": avg_latency,
            "memory_mb": 0.0,
            "query_count": self.query_count,
            "refusal_rate": self.refusals / max(1, self.query_count),
            "intent_confidence": 0.8,
            "query_length": 50,
            "available_chunks": 1000,
            "last_evidence_score": 0.7,
            "contradiction_rate": 0.0,
            "topic_diversity": 0.5,
            "time_since_last_query_s": 0.0,
        }

    def _create_cached_result(self, chunk_ids) -> RetrieveResult:
        """Create result from cached chunk IDs"""
        from core.contracts.retrieved_chunk import RetrievedChunk

        chunks_list = []
        for i, cid in enumerate(chunk_ids):
            if cid in self.retrieval_engine.chunks:
                chunk_data = self.retrieval_engine.chunks[cid]
                chunks_list.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=chunk_data.text,
                        topic_id=0,
                        similarity_score=0.9,
                        chunk_index=i,
                        source_doc=chunk_data.source_doc,
                    )
                )

        return RetrieveResult(
            chunks=tuple(chunks_list),
            from_cache=True,
            total_searched=0,
            retrieval_time_ms=0.0,
        )

    def _update_memory_gate(self, plan, retrieval_result, validation):
        """Update cache and beliefs"""

        if validation.is_valid:
            chunk_ids = {chunk.chunk_id for chunk in retrieval_result.chunks}
            self.mutation_gate.admit_to_cache(plan, chunk_ids, validation)
            self.mutation_gate.create_beliefs(validation)

        if validation.contradictions:
            self.mutation_gate.handle_contradiction(validation)

        self.mutation_gate.evict_expired()

    def _train_rl(self, context, validation, latency_ms):
        """Train RL agent"""

        state = self.rl_agent.extract_state_features(context)

        reward = self.rl_agent.compute_reward(
            correct=validation.is_valid,
            latency_ms=latency_ms,
            memory_mb=0.0,
            hallucinated=validation.status.value == "insufficient_evidence",
        )

        loss = self.rl_agent.train_step()
        if loss is not None:
            self.logger.debug(f"RL training loss: {loss:.4f}")

    def _create_success_response(
        self, answer, validation, retrieval_result, latency_ms
    ):
        """Create successful response"""
        self.total_latency += latency_ms

        return {
            "status": "success",
            "answer": answer,
            "evidence_score": validation.evidence_score,
            "num_chunks": retrieval_result.num_chunks,
            "from_cache": retrieval_result.from_cache,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

    def _create_refusal_response(self, reason, start_time):
        """Create refusal response"""
        latency_ms = (time.time() - start_time) * 1000
        self.total_latency += latency_ms

        return {
            "status": "refused",
            "reason": reason,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

    def _create_validation_refusal(self, validation, start_time):
        """Create validation refusal response"""
        latency_ms = (time.time() - start_time) * 1000
        self.total_latency += latency_ms

        return {
            "status": "refused",
            "reason": validation.reasoning,
            "evidence_score": validation.evidence_score,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

    def _create_error_response(self, error_msg, start_time):
        """Create error response"""
        latency_ms = (time.time() - start_time) * 1000

        return {
            "status": "error",
            "error": error_msg,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down pipeline...")

        self.mutation_gate.persist()
        self.rl_agent.save_model()

        self.logger.info(f"Total queries: {self.query_count}")
        self.logger.info(
            f"Cache hit rate: {self.cache_hits / max(1, self.query_count):.2%}"
        )
        self.logger.info(
            f"Refusal rate: {self.refusals / max(1, self.query_count):.2%}"
        )
        self.logger.info(
            f"Avg latency: {self.total_latency / max(1, self.query_count):.1f}ms"
        )

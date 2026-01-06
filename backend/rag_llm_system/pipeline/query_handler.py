import time
from typing import Dict, Optional, Tuple

from cache.cache_key import CacheKey
from cache.cache_manager import CacheManager
from config.thresholds import THRESHOLDS, Query
from ingest.embedder import Embedder
from ingest.indexer import FAISSIndexer
from memory.belief_store import BeliefStore
from memory.episodic import EpisodicMemory
from memory.stm import ShortTermMemory
from memory.wm import WorkingMemory
from prompt.context_selector import ContextSelector
from prompt.formatter import PromptFormatter
from retrieval.intend_detector import IntentDetector
from retrieval.reranker import Reranker
from retrieval.retriver import Retriever
from retrieval.topic_router import TopicRouter
from rl.policy import Action, RLPolicy
from rl.reward import RewardComputer
from rl.state import StateBuilder
from validation.claim_extractor import ClaimExtractor
from validation.contradiction_detector import ContradictionDetector
from validation.evidence_alignment import EvidenceAligner
from validation.refusal_logic import RefusalLogic


class QueryHandler:
    """Main query processing pipeline."""

    def __init__(
        self,
        embedder: Embedder,
        indexer: FAISSIndexer,
        topic_router: TopicRouter,
        cache_manager: CacheManager,
        stm: ShortTermMemory,
        wm: WorkingMemory,
        episodic: EpisodicMemory,
        belief_store: BeliefStore,
        rl_policy: RLPolicy,
    ):
        self.embedder = embedder
        self.indexer = indexer
        self.topic_router = topic_router
        self.cache_manager = cache_manager
        self.stm = stm
        self.wm = wm
        self.episodic = episodic
        self.belief_store = belief_store
        self.rl_policy = rl_policy

        # Sub-modules
        self.retriever = Retriever(indexer, embedder, topic_router)
        self.reranker = Reranker()
        self.intent_detector = IntentDetector()
        self.claim_extractor = ClaimExtractor()
        self.evidence_aligner = EvidenceAligner(embedder)
        self.contradiction_detector = ContradictionDetector(belief_store)
        self.context_selector = ContextSelector()
        self.prompt_formatter = PromptFormatter()
        self.refusal_logic = RefusalLogic()

    def process_query(self, query_text: str) -> Tuple[str, Dict]:
        """Execute full pipeline."""
        start_time = time.time()
        metadata = {"steps": []}

        # Step 1: Topic & Intent Detection
        query_emb = self.embedder.embed_query(query_text)
        topic_id = self.topic_router.predict_topic(query_emb)
        intent = self.intent_detector.detect(query_text)

        query = Query(text=query_text, topic_id=topic_id, intent=intent)
        metadata["steps"].append("topic_intent_detection")

        # Step 2: Build RL State
        cache_key = CacheKey.create(topic_id, intent, query_text)
        cached_chunks = self.cache_manager.lookup(cache_key)
        cache_hit = cached_chunks is not None

        state = StateBuilder.build(
            topic_id=topic_id,
            intent=intent,
            cache_hit=cache_hit,
            last_evidence_score=0.0,
            contradiction_flag=False,
            token_budget=2048,
        )
        metadata["steps"].append("rl_state_build")

        # Step 3: Policy Decision
        action = self.rl_policy.decide_action(state)
        metadata["action"] = action.value

        if action == Action.REFUSE:
            return self._create_refusal(), metadata

        # Step 4: Retrieval (if needed)
        chunks = []
        if action in [Action.RETRIEVE_LTM, Action.RETRIEVE_EM]:
            if cache_hit:
                # Get from cache
                chunk_ids = cached_chunks
                chunks = [
                    self.indexer.chunk_map[cid]
                    for cid in chunk_ids
                    if cid in self.indexer.chunk_map
                ]
                metadata["steps"].append("cache_hit")
            else:
                # Retrieve from index
                chunks = self.retriever.retrieve(query, k=10)
                chunks = self.reranker.rerank(chunks, query)

                # Cache results
                chunk_ids = [c.chunk_id for c in chunks]
                self.cache_manager.store(cache_key, chunk_ids, score=0.8)
                metadata["steps"].append("ann_retrieval")

        # Step 5: Context Assembly
        beliefs = list(self.belief_store.beliefs.values())[:5]
        context = self.context_selector.select_context(
            self.stm, self.wm, chunks, beliefs
        )
        prompt = self.prompt_formatter.format(query_text, context)
        metadata["steps"].append("prompt_assembly")

        # Step 6: LLM Generation (mock)
        response_text = self._generate_response(prompt)
        metadata["steps"].append("llm_generation")

        # Step 7: Validation
        claims = self.claim_extractor.extract(response_text, intent)
        evidence_score = self.evidence_aligner.compute_alignment(claims, chunks)
        has_contradiction = self.contradiction_detector.detect(claims)

        metadata["evidence_score"] = evidence_score
        metadata["has_contradiction"] = has_contradiction
        metadata["steps"].append("validation")

        # Step 8: Refusal Check
        should_refuse = self.refusal_logic.should_refuse(
            evidence_score=evidence_score,
            has_contradiction=has_contradiction,
            retrieval_failed=len(chunks) == 0 and action != Action.SKIP_RETRIEVAL,
        )

        if should_refuse:
            return self._create_refusal(), metadata

        # Step 9: Update Memories
        self.stm.add_turn(query_text, response_text)

        for claim in claims:
            if evidence_score > THRESHOLDS.MIN_BELIEF_CONFIDENCE:
                evidence_refs = [c.chunk_id for c in chunks]
                self.belief_store.add_belief(claim, evidence_refs, evidence_score)

        metadata["steps"].append("belief_update")

        # Step 10: Compute Reward
        latency_ms = (time.time() - start_time) * 1000
        reward = RewardComputer.compute(
            evidence_score=evidence_score,
            is_consistent=not has_contradiction,
            cache_hit=cache_hit,
            has_hallucination=evidence_score < 0.3,
            has_contradiction=has_contradiction,
            did_retrieval=len(chunks) > 0,
            latency_ms=latency_ms,
            tokens_used=len(prompt.split()),
            token_budget=2048,
        )

        metadata["reward"] = reward
        metadata["latency_ms"] = latency_ms
        metadata["steps"].append("rl_reward")

        return response_text, metadata

    def _generate_response(self, prompt: str) -> str:
        """Mock LLM generation."""
        # In production, this would call actual LLM API
        return "This is a mock response based on the provided evidence."

    def _create_refusal(self) -> str:
        """Create refusal message."""
        return "I cannot provide a confident answer based on available evidence."

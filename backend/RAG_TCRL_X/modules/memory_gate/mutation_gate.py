import pickle
from typing import Dict, List, Set, Optional, FrozenSet
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from core.contracts.retrieval_plan import RetrievalPlan
from core.contracts.validation import Validation
from logger import Logger
from config import Config


@dataclass
class CachedPlan:
    """Cached retrieval plan with metadata"""

    plan: RetrievalPlan
    chunk_ids: FrozenSet[int]
    evidence_score: float
    hit_count: int
    last_access: datetime
    created_at: datetime

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        age = datetime.now() - self.created_at
        return age > timedelta(seconds=Config.CACHE_TTL_SECONDS)


@dataclass
class Belief:
    """System belief with evidence tracking"""

    claim: str
    evidence_chunk_ids: FrozenSet[int]
    confidence: float
    created_at: datetime
    last_updated: datetime
    revision_count: int = 0
    status: str = "active"  # active, revised, invalidated

    def decay_confidence(self):
        """Decay confidence due to contradiction"""
        self.confidence = max(
            Config.MIN_CONFIDENCE, self.confidence - Config.CONFIDENCE_DECAY
        )
        self.revision_count += 1
        self.last_updated = datetime.now()


class MutationGate:
    """Manages cache and belief system with invalidation"""

    def __init__(self, cache_path: Path, beliefs_path: Path):
        self.cache_path = cache_path
        self.beliefs_path = beliefs_path
        self.logger = Logger().get_logger("MutationGate")

        # In-memory stores
        self.cache: Dict[str, CachedPlan] = {}
        self.beliefs: Dict[str, Belief] = {}

        # Load from disk
        self._load()

    def check_cache(self, plan: RetrievalPlan) -> Optional[FrozenSet[int]]:
        """Check cache for retrieval plan"""
        if not plan.use_cache:
            return None

        cache_key = plan.cache_key

        if cache_key in self.cache:
            cached = self.cache[cache_key]

            # Check expiry
            if cached.is_expired():
                self.logger.debug(f"Cache expired: {cache_key}")
                del self.cache[cache_key]
                return None

            # Update access metadata
            cached.hit_count += 1
            cached.last_access = datetime.now()

            self.logger.debug(f"Cache hit: {cache_key} (hits={cached.hit_count})")
            return cached.chunk_ids

        self.logger.debug(f"Cache miss: {cache_key}")
        return None

    def admit_to_cache(
        self, plan: RetrievalPlan, chunk_ids: Set[int], validation: Validation
    ):
        """Admit retrieval plan to cache if it meets criteria"""

        # Check admission criteria
        if validation.evidence_score < Config.CACHE_ADMISSION_THRESHOLD:
            self.logger.debug("Cache admission rejected: low evidence score")
            return

        cache_key = plan.cache_key

        # Check if already exists
        if cache_key in self.cache:
            existing = self.cache[cache_key]
            existing.hit_count += 1
            existing.last_access = datetime.now()
            return

        # Create new entry
        cached = CachedPlan(
            plan=plan,
            chunk_ids=frozenset(chunk_ids),
            evidence_score=validation.evidence_score,
            hit_count=1,
            last_access=datetime.now(),
            created_at=datetime.now(),
        )

        self.cache[cache_key] = cached
        self.logger.info(f"Admitted to cache: {cache_key}")

    def create_beliefs(self, validation: Validation):
        """Create beliefs from validated claims"""
        if not validation.is_valid:
            return

        for claim in validation.claims:
            belief_key = self._hash_claim(claim)

            if belief_key in self.beliefs:
                # Update existing belief
                belief = self.beliefs[belief_key]
                belief.last_updated = datetime.now()
                belief.confidence = min(1.0, belief.confidence + 0.1)
            else:
                # Create new belief
                belief = Belief(
                    claim=claim,
                    evidence_chunk_ids=validation.evidence_chunk_ids,
                    confidence=Config.INITIAL_CONFIDENCE,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                )
                self.beliefs[belief_key] = belief

        self.logger.debug(f"Created/updated {len(validation.claims)} beliefs")

    def handle_contradiction(self, validation: Validation):
        """Handle contradiction by revising beliefs and invalidating cache"""
        if not validation.contradictions:
            return

        # Revise related beliefs
        for claim in validation.claims:
            belief_key = self._hash_claim(claim)
            if belief_key in self.beliefs:
                belief = self.beliefs[belief_key]
                belief.decay_confidence()

                if belief.confidence <= Config.MIN_CONFIDENCE:
                    belief.status = "invalidated"
                    self.logger.warning(f"Invalidated belief: {claim[:50]}...")
                else:
                    belief.status = "revised"
                    self.logger.info(f"Revised belief: {claim[:50]}...")

        # Invalidate related cache entries
        invalidated = 0
        for cache_key, cached in list(self.cache.items()):
            # Check if cached chunks overlap with contradicted evidence
            if cached.chunk_ids & validation.evidence_chunk_ids:
                del self.cache[cache_key]
                invalidated += 1

        self.logger.info(
            f"Invalidated {invalidated} cache entries due to contradiction"
        )

    def evict_expired(self):
        """Evict expired cache entries"""
        expired_keys = [
            key for key, cached in self.cache.items() if cached.is_expired()
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.info(f"Evicted {len(expired_keys)} expired cache entries")

    def persist(self):
        """Persist cache and beliefs to disk"""
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

        with open(self.beliefs_path, "wb") as f:
            pickle.dump(self.beliefs, f)

        self.logger.debug("Persisted cache and beliefs")

    def _load(self):
        """Load cache and beliefs from disk"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.cache)} cache entries")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        if self.beliefs_path.exists():
            try:
                with open(self.beliefs_path, "rb") as f:
                    self.beliefs = pickle.load(f)
                self.logger.info(f"Loaded {len(self.beliefs)} beliefs")
            except Exception as e:
                self.logger.warning(f"Failed to load beliefs: {e}")

    @staticmethod
    def _hash_claim(claim: str) -> str:
        """Generate hash for claim"""
        import hashlib

        return hashlib.sha256(claim.encode()).hexdigest()[:16]

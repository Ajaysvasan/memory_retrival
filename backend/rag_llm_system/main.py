import pickle
from typing import Dict, List, Tuple

import numpy as np

from cache.cache_manager import CacheManager
from ingest.chunker import Chunker
from ingest.dedup import Deduplicator
from ingest.embedder import Embedder
from ingest.indexer import FAISSIndexer
from ingest.loader import DocumentLoader
from memory.belief_store import BeliefStore
from memory.episodic import EpisodicMemory
from memory.stm import ShortTermMemory
from memory.wm import WorkingMemory
from pipeline.query_handler import QueryHandler
from retrieval.topic_router import TopicRouter
from rl.policy import RLPolicy


class RAGSystem:
    """Complete RAG-LLM system."""

    def __init__(self):
        self.embedder = Embedder()
        self.indexer = FAISSIndexer()
        self.topic_router = TopicRouter()
        self.cache_manager = CacheManager()

        # Memory systems
        self.stm = ShortTermMemory()
        self.wm = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.belief_store = BeliefStore()

        # RL
        self.rl_policy = RLPolicy()

        # Pipeline
        self.query_handler = QueryHandler(
            embedder=self.embedder,
            indexer=self.indexer,
            topic_router=self.topic_router,
            cache_manager=self.cache_manager,
            stm=self.stm,
            wm=self.wm,
            episodic=self.episodic,
            belief_store=self.belief_store,
            rl_policy=self.rl_policy,
        )

    def ingest_documents(self, filepaths: List[str]):
        """Ingest and index documents."""
        loader = DocumentLoader()
        chunker = Chunker()
        deduplicator = Deduplicator()

        all_chunks = []

        for filepath in filepaths:
            # Load
            paragraphs = list(loader.load_document(filepath))

            # Chunk
            chunks = chunker.chunk_document(paragraphs)
            all_chunks.extend(chunks)

        # Deduplicate
        all_chunks = deduplicator.deduplicate_chunks(all_chunks)

        # Embed
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)

        # Store embeddings in chunks
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # Build index
        self.indexer.build_index(all_chunks, embeddings)

        # Fit topic router
        self.topic_router.fit(embeddings)

        print(f"Indexed {len(all_chunks)} chunks from {len(filepaths)} documents")

    def query(self, text: str) -> Tuple[str, Dict]:
        """Process a query."""
        return self.query_handler.process_query(text)

    def save(self, path: str):
        """Save system state."""
        import os

        os.makedirs(path, exist_ok=True)

        self.indexer.save(f"{path}/index")

        with open(f"{path}/topic_router.pkl", "wb") as f:
            pickle.dump(self.topic_router, f)

        with open(f"{path}/belief_store.pkl", "wb") as f:
            pickle.dump(self.belief_store, f)

    def load(self, path: str):
        """Load system state."""
        self.indexer.load(f"{path}/index")

        with open(f"{path}/topic_router.pkl", "rb") as f:
            self.topic_router = pickle.load(f)

        with open(f"{path}/belief_store.pkl", "rb") as f:
            self.belief_store = pickle.load(f)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_basic_usage():
    """Basic usage example."""

    # Initialize system
    system = RAGSystem()

    # Ingest documents
    documents = ["data/raw/doc1.txt", "data/raw/doc2.txt", "data/raw/doc3.txt"]

    print("Ingesting documents...")
    system.ingest_documents(documents)

    # Process queries
    queries = [
        "What is the main topic of the documents?",
        "Can you explain the key concepts?",
        "How does this relate to previous discussions?",
    ]

    for query_text in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        print("=" * 60)

        response, metadata = system.query(query_text)

        print(f"\nResponse: {response}")
        print(f"\nMetadata:")
        print(f"  - Action: {metadata.get('action')}")
        print(f"  - Evidence Score: {metadata.get('evidence_score', 0):.3f}")
        print(f"  - Contradiction: {metadata.get('has_contradiction', False)}")
        print(f"  - Latency: {metadata.get('latency_ms', 0):.2f}ms")
        print(f"  - Reward: {metadata.get('reward', 0):.3f}")
        print(f"  - Pipeline Steps: {' -> '.join(metadata.get('steps', []))}")

    # Save system state
    print("\n\nSaving system state...")
    system.save("checkpoints/system_v1")
    print("Done!")


def example_advanced_usage():
    """Advanced usage with constraints and episodic memory."""

    system = RAGSystem()
    system.ingest_documents(["data/raw/technical_docs.txt"])

    # Add working memory constraints
    system.wm.add_constraint("Focus on technical accuracy")
    system.wm.add_constraint("Provide code examples when relevant")

    # Simulate multi-turn conversation
    conversation = [
        "What are the key architectural components?",
        "How does the caching system work?",
        "Can you show an example of the retrieval process?",
        "What are the performance implications?",
    ]

    for i, query in enumerate(conversation, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {query}")
        print("=" * 60)

        response, metadata = system.query(query)
        print(f"Response: {response}")

        # Store episode summary after each turn
        if i % 2 == 0:
            summary = f"Discussed: {query[:50]}..."
            system.episodic.add_episode(summary, metadata)


def example_monitoring():
    """Example with detailed monitoring."""

    system = RAGSystem()
    system.ingest_documents(["data/raw/docs.txt"])

    # Track metrics
    metrics = {
        "total_queries": 0,
        "cache_hits": 0,
        "refusals": 0,
        "avg_evidence_score": [],
        "avg_latency": [],
    }

    queries = ["Query 1", "Query 2", "Query 3"] * 10

    for query in queries:
        response, metadata = system.query(query)

        metrics["total_queries"] += 1

        if metadata.get("action") == "refuse":
            metrics["refusals"] += 1

        if "cache_hit" in metadata.get("steps", []):
            metrics["cache_hits"] += 1

        if "evidence_score" in metadata:
            metrics["avg_evidence_score"].append(metadata["evidence_score"])

        if "latency_ms" in metadata:
            metrics["avg_latency"].append(metadata["latency_ms"])

    # Print summary
    print("\n" + "=" * 60)
    print("SYSTEM METRICS")
    print("=" * 60)
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Cache Hit Rate: {metrics['cache_hits']/metrics['total_queries']*100:.1f}%")
    print(f"Refusal Rate: {metrics['refusals']/metrics['total_queries']*100:.1f}%")
    print(f"Avg Evidence Score: {np.mean(metrics['avg_evidence_score']):.3f}")
    print(f"Avg Latency: {np.mean(metrics['avg_latency']):.2f}ms")
    print(
        f"Cache Stats: L1 hits={system.cache_manager.l1.hits}, misses={system.cache_manager.l1.misses}"
    )


# ============================================================================
# TESTING UTILITIES
# ============================================================================


def create_sample_documents():
    """Create sample documents for testing."""
    import os

    os.makedirs("data/raw", exist_ok=True)

    docs = {
        "data/raw/doc1.txt": """
            Machine learning is a subset of artificial intelligence.
            It focuses on building systems that learn from data.
            Common techniques include supervised learning, unsupervised learning, and reinforcement learning.
            Neural networks are a popular approach in modern machine learning.
        """,
        "data/raw/doc2.txt": """
            Retrieval-augmented generation combines information retrieval with language generation.
            This approach helps reduce hallucinations in language models.
            By grounding responses in retrieved evidence, systems can be more factual.
            Vector databases are commonly used for efficient similarity search.
        """,
        "data/raw/doc3.txt": """
            Caching strategies are important for system performance.
            Multi-tier caches can balance speed and capacity.
            LRU eviction policies are simple and effective.
            Admission control prevents cache pollution.
        """,
    }

    for filepath, content in docs.items():
        with open(filepath, "w") as f:
            f.write(content.strip())

    print("Created sample documents in data/raw/")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main execution."""
    print("RAG-LLM System")
    print("=" * 60)

    # Create sample documents
    create_sample_documents()

    # Run basic example
    print("\n1. Running basic usage example...")
    example_basic_usage()

    # Uncomment to run other examples:
    print("\n2. Running advanced usage example...")
    example_advanced_usage()

    print("\n3. Running monitoring example...")
    example_monitoring()


if __name__ == "__main__":
    main()

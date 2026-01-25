import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.lifecycle.system_gate import SystemGate
from data.initialization import IntegrityValidator, SystemInitializer
from data.retrieval_engine import RetrievalEngine
from logger import Logger
from modules.generation.generator_adaptor import GeneratorAdaptor
from modules.intent.heuristic_intent_classifier import HeuristicIntentClassifier
from modules.memory_gate.mutation_gate import MutationGate
from modules.planning.retrival_planner import RetrievalPlanner
from modules.rl.rl_agent import RLAgent
from modules.validation.validator import Validator
from orchestration.phase_a_orchestrator import PhaseAOrchestrator
from orchestration.pipeline import Pipeline


def Model(query_text: str = ""):

    # Initialize logger first
    logger = Logger().get_logger("Main")

    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      RAG-TCRL-X v1.0                         ║
    ║   Topic-Conditioned RAG with RL Control & Belief System      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        # Step 1: Validate configuration
        logger.info("Validating configuration...")
        Config.validate()
        logger.info("✓ Configuration valid")

        # Step 2: Check initialization requirements
        system_gate = SystemGate(Config)
        system_gate.validate_runtime_requirements()

        needs_init = system_gate.check_initialization_required()

        # Step 3: Initialize or load system
        initializer = SystemInitializer()

        if needs_init:
            logger.info("Performing full system initialization...")
            chunks, embedding_engine, faiss_indexer = initializer.initialize()
            system_gate.save_version()
        else:
            logger.info("Loading existing system state...")
            chunks, embedding_engine, faiss_indexer = initializer.load_existing()
        # Step 4: Validate integrity
        validator_integrity = IntegrityValidator()
        validator_integrity.validate(chunks, embedding_engine, faiss_indexer)

        # Step 5: Build components
        logger.info("Building system components...")

        # Intent classification
        intent_classifier = HeuristicIntentClassifier()

        # Retrieval planning
        retrieval_planner = RetrievalPlanner(num_topics=Config.NUM_TOPICS)

        # Retrieval engine
        retrieval_engine = RetrievalEngine(
            embedding_engine=embedding_engine,
            faiss_indexer=faiss_indexer,
            chunks=chunks,
        )

        # Validation
        validator = Validator(embedding_engine=embedding_engine)

        # Generation
        generator = GeneratorAdaptor()

        # Memory gate (cache + beliefs)
        mutation_gate = MutationGate(
            cache_path=Config.CACHE_PATH, beliefs_path=Config.BELIEFS_PATH
        )

        # RL agent
        rl_agent = RLAgent(model_path=Config.RL_MODEL_PATH)

        # Phase A orchestrator
        phase_a_orchestrator = PhaseAOrchestrator(
            intent_classifier=intent_classifier,
            retrieval_planner=retrieval_planner,
            rl_agent=rl_agent,
        )

        # Main pipeline
        pipeline = Pipeline(
            phase_a_orchestrator=phase_a_orchestrator,
            retrieval_engine=retrieval_engine,
            validator=validator,
            generator=generator,
            mutation_gate=mutation_gate,
            rl_agent=rl_agent,
        )

        logger.info("✓ System components initialized")

        # Step 6: Run demo queries
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING DEMO QUERIES")
        logger.info("=" * 80 + "\n")

        demo_queries = [
            "What is the main topic of the documents?",
            "Explain the key concepts discussed.",
            "Compare the different approaches mentioned.",
            "What are the procedural steps involved?",
        ]

        for query_text in demo_queries:
            response = pipeline.process(query_text)
            print_response(response)

        # Step 7: Interactive mode
        logger.info("\n" + "=" * 80)
        logger.info("ENTERING INTERACTIVE MODE")
        logger.info("=" * 80)
        logger.info("Enter queries (or 'quit' to exit)")
        logger.info("=" * 80 + "\n")

        while True:
            query_text = input("Enter the query: ")
            try:
                if query_text.lower() in ["quit", "exit", "q"]:
                    break

                if not query_text:
                    continue

                response = pipeline.process(query_text)
                print_response(response)

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)

        try:
            response = pipeline.process(query_text)
            print(response["answer"])
            return {"answer": response["answer"]}
        except Exception as e:
            logger.error(f"Error in processing the query {e}", exc_info=True)
            raise Exception(f"Error in processing the query {e}")
        # # Step 8: Shutdown
        # logger.info("\n" + "=" * 80)
        # logger.info("SHUTTING DOWN")
        # logger.info("=" * 80)

        # pipeline.shutdown()

        # logger.info("✓ Shutdown complete")
        # logger.info("Goodbye!")

    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


def print_response(response: dict):
    """Pretty print response"""
    print("\n" + "-" * 80)

    if response["status"] == "success":
        print(f"Status: ✓ SUCCESS")
        print(f"Answer: {response['answer']}")
        print(f"Evidence Score: {response['evidence_score']:.2f}")
        print(f"Chunks Used: {response['num_chunks']}")
        print(f"From Cache: {response['from_cache']}")
        print(f"Latency: {response['latency_ms']:.1f}ms")

    elif response["status"] == "refused":
        print(f"Status: ⚠ REFUSED")
        print(f"Reason: {response['reason']}")
        print(f"Latency: {response['latency_ms']:.1f}ms")

    else:
        print(f"Status: ✗ ERROR")
        print(f"Error: {response.get('error', 'Unknown error')}")
        print(f"Latency: {response.get('latency_ms', 0):.1f}ms")

    print("-" * 80)


try:
    if __name__ == "__main__":
        Model("")
except:
    pass

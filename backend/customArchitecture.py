import os
import sys

backend_dir = os.path.dirname(os.path.abspath(__file__))

rag_dir = os.path.join(backend_dir, "RAG_TCRL_X")

sys.path.append(backend_dir)
sys.path.append(rag_dir)
sys.path.append(rag_dir)
from RAG_TCRL_X.config import Config
from RAG_TCRL_X.core.lifecycle.system_gate import SystemGate
from RAG_TCRL_X.data.initialization import IntegrityValidator, SystemInitializer
from RAG_TCRL_X.data.retrieval_engine import RetrievalEngine
from RAG_TCRL_X.logger import Logger
from RAG_TCRL_X.Model import print_response
from RAG_TCRL_X.modules.generation.generator_adaptor import GeneratorAdaptor
from RAG_TCRL_X.modules.intent.heuristic_intent_classifier import (
    HeuristicIntentClassifier,
)
from RAG_TCRL_X.modules.memory_gate.mutation_gate import MutationGate
from RAG_TCRL_X.modules.planning.retrival_planner import RetrievalPlanner
from RAG_TCRL_X.modules.rl.rl_agent import RLAgent
from RAG_TCRL_X.modules.validation.validator import Validator
from RAG_TCRL_X.orchestration.phase_a_orchestrator import PhaseAOrchestrator
from RAG_TCRL_X.orchestration.pipeline import Pipeline


def load():
    pass


logger = Logger().get_logger("Main")
logger.info("starting the application")
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
except Exception as e:
    logger.error(
        f"The following exception occured while trying to initialize the model {e}"
    )


def process():
    pass

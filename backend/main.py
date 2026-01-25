# this is going to contain only the route that is neccessary for the model services that is only one route as I don't want to re write the entire vibe coded code base
# which is such a pain

# Note to self
# even in the future don't attempt to refactor this code base


import os
import sys
import threading
import time

from huggingface_hub import TextToSpeechEarlyStoppingEnum

# this note is for self
# You shouldn't have chose to do this


# --- PATH SETUP START ---
# 1. Get the path to the 'backend' folder
backend_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the 'RAG_TCRL_X' folder
rag_dir = os.path.join(backend_dir, "RAG_TCRL_X")
test_bench = os.path.join(backend_dir, "test_bench")

# 3. Add BOTH to sys.path

sys.path.insert(0, rag_dir)
sys.path.append(backend_dir)  # Allows "from RAG_TCRL_X import ..."
sys.path.append(test_bench)
# --- PATH SETUP END ---

# Now your imports will work without changing the inner code
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from test_bench.orchestrator import TestBenchOrchestrator

# adding all the packages so that I can import what I want to use
module_dir = os.path.join(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)
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

# Add project root to path

app = FastAPI(title="Memory retrival model")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserQuery(BaseModel):
    query: str


# These are the model initialization
# This will be removed why right now the model is getting loaded directly into the RAM which is not what we wanted

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

# initialization of all the other three models
print("Testing if this initialization works")
testBench = TestBenchOrchestrator()
testBench.train_all()
print("it worked")

# I didn't expect that to work lol


@app.post("/api/chat/")
def query(query_text: UserQuery):
    overAllReponse = dict()
    try:
        response = pipeline.process(query_text.query)
        # I still need cache hit , accuracy , retrieved chunks and cache hit
        # check the status of the response if it is success then return the output
        # if it is other than success i.e refused or erorr then return what the message is
        if response["status"] == "success":
            overAllReponse["model_one_answer"] = response["answer"]
            overAllReponse["model_one_latency"] = response["latency_ms"]
        elif response["status"] == "refused":
            overAllReponse["model_one_answer"] = response["reason"]
            overAllReponse["model_one_latency"] = response["latency_ms"]

        else:
            overAllReponse["model_one_answer"] = (
                f"The following error occured {response.get('error', 'Unknown error')}"
            )
            overAllReponse["model_one_latency"] = response["latency_ms"]
        response = testBench.arch1.query(query_text.query)
        overAllReponse["model_two_answer"] = response.output
        overAllReponse["model_two_latency"] = response.latency
        response = testBench.arch2.query(query_text.query)
        overAllReponse["model_three_answer"] = response.output
        overAllReponse["model_three_latency"] = response.latency
        response = testBench.arch3.query(query_text.query)
        overAllReponse["model_four_answer"] = response.output
        overAllReponse["model_four_latency"] = response.latency

        return overAllReponse
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


if __name__ == "__main__":
    import uvicorn

    # # This runs the server if you type `python main.py`
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        # It workeddddd wow
    except KeyboardInterrupt:
        pipeline.shutdown()
        pass

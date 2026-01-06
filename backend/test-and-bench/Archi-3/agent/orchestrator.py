from agent.registry import ToolRegistry


class AgentOrchestrator:
    def __init__(self):
        self.registry = ToolRegistry()

    def register(self, tool):
        self.registry.register(tool)

    def run(self, query):
        # Step 1: Retrieval
        retrieved_docs = self.registry.get("hybrid_search").execute(query)

        # Step 2: Relevance validation (NOW USED)
        relevance_result = self.registry.get("check_relevance").execute(
            query=query,
            documents=retrieved_docs
        )

        # If documents are not relevant, stop early
        if not relevance_result["is_relevant"]:
            return (
                "Retrieved documents are not sufficiently relevant.",
                retrieved_docs,
            )

        # Step 3: Answer synthesis
        answer = self.registry.get("synthesize_answer").execute(
            query=query,
            documents=retrieved_docs
        )

        return answer, retrieved_docs

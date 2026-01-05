class AgentOrchestrator:
    def __init__(self, retriever, synthesizer):
        self.retriever = retriever
        self.synthesizer = synthesizer

    def run(self, query):
        docs = self.retriever.execute(query)
        answer = self.synthesizer.execute(query, docs)
        return answer, docs

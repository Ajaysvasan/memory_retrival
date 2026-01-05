class AnswerSynthesisTool:
    def execute(self, query, documents):
        return " ".join(d.content for d in documents[:2])

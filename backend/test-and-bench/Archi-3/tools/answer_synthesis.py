from tools.base import BaseTool

class AnswerSynthesisTool(BaseTool):
    def __init__(self):
        super().__init__("synthesize_answer")

    def execute(self, query, documents):
        return " ".join(d.content for d in documents[:2])

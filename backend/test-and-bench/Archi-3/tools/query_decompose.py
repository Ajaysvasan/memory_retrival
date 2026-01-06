from tools.base import BaseTool

class QueryDecompositionTool(BaseTool):
    def __init__(self):
        super().__init__("decompose_query")

    def execute(self, query):
        return [query]

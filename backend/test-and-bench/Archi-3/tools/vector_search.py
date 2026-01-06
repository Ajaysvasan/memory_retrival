from tools.base import BaseTool

class VectorDBSearchTool(BaseTool):
    def __init__(self, vector_db):
        super().__init__("vector_search")
        self.db = vector_db

    def execute(self, query, top_k=5):
        return self.db.search(query, top_k)

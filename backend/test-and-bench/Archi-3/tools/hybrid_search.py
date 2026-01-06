from tools.base import BaseTool

class HybridSearchTool(BaseTool):
    def __init__(self, vector_tool, bm25_tool, alpha=0.6):
        super().__init__("hybrid_search")
        self.vector = vector_tool
        self.bm25 = bm25_tool
        self.alpha = alpha

    def execute(self, query, top_k=5):
        v = self.vector.execute(query, top_k*2)
        b = self.bm25.execute(query, top_k*2)
        scores = {}

        for d in v:
            scores[d.doc_id] = self.alpha * d.retrieval_score
        for d in b:
            scores[d.doc_id] = scores.get(d.doc_id, 0) + (1-self.alpha)*d.retrieval_score

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        docs = {d.doc_id: d for d in v+b}
        return [docs[i] for i, _ in merged]

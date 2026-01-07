import json
from core.document import Document
from rag.fid_rag_system import FiDRAGwithVectorDB

docs = [
    Document("d1", "Neural networks learn via backpropagation."),
    Document("d2", "Deep learning uses layered neural networks."),
]

rag = FiDRAGwithVectorDB()
rag.setup(docs)

query = "How do neural networks learn?"
human = "They learn using backpropagation."

res = rag.evaluate(query, human)

print(json.dumps({
    "query": res.query,
    "overall_consistency": float(res.consistency.overall_consistency),
    "agreement": float(res.comparison.agreement_score),
    "recommendation": res.comparison.recommendation,
    "execution_time": float(res.total_time)
}, indent=2))

import json
from core.document import Document
from rag.enhanced_rag import HybridRAGwithVectorDB

docs = [
    Document("d1","Neural networks learn via backpropagation."),
    Document("d2","Deep learning uses layered neural networks."),
    Document("d3","Gradient descent updates weights."),
]

rag = HybridRAGwithVectorDB()
rag.setup(docs)

query = "How do neural networks learn?"
human = "Neural networks learn by backpropagation."

res = rag.evaluate(query, human)

print(json.dumps({
    "query": res.query,
    "overall_consistency": float(res.consistency.overall_consistency),
    "agreement": float(res.comparison.agreement_score),
    "recommendation": res.comparison.recommendation,
    "execution_time": float(res.total_time)
}, indent=2))

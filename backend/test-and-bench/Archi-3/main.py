import json
from core.document import Document
from system.agentic_rag_system import AgenticRAGSystem

documents = [
    Document("d1", "Neural networks learn via backpropagation.", {}),
    Document("d2", "Gradient descent updates weights.", {}),
]

rag = AgenticRAGSystem(documents)

query = "How do neural networks learn?"
human = "Neural networks learn through backpropagation."

result = rag.run(query, human)

output = {
    "query": result.query,
    "overall_consistency": float(result.consistency.overall_consistency),
    "agreement": float(result.comparison.agreement_score),
    "recommendation": result.comparison.recommendation,
    "execution_time": float(result.total_time),
}

print(json.dumps(output, indent=2))

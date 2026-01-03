from core.document import Document
from rag.fid_rag_system import FusionInDecoderRAG

documents = [
    Document("d1", "Neural networks learn using backpropagation."),
    Document("d2", "Deep learning uses layered neural networks."),
]

rag = FusionInDecoderRAG(use_fusion=True)
rag.index(documents)

query = "How do neural networks learn?"
human_answer = "Neural networks learn by adjusting weights using backpropagation."

result = rag.evaluate_full(query, human_answer)

print(result.generated_answer)
print(result.comparison.recommendation)

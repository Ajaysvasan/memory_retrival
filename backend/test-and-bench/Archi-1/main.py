import json
from core.document import Document
from rag.enhanced_rag import EnhancedRAGSystem


def main():
    documents = [
        Document("doc1", "Machine learning enables systems to learn from data."),
        Document("doc2", "Deep learning uses neural networks with many layers."),
        Document("doc3", "Neural networks are inspired by biological neurons."),
        Document("doc4", "NLP helps computers understand human language."),
        Document("doc5", "Transformers rely on self-attention mechanisms."),
    ]

    rag = EnhancedRAGSystem(alpha=0.6, rerank_top_k=3)
    rag.index(documents)

    query = "How do neural networks learn?"
    ai_answer = "Neural networks learn using backpropagation and layered representations."
    human_answer = "Neural networks learn by adjusting weights through backpropagation."

    result = rag.evaluate_full(query, ai_answer, human_answer, k=5)

    print(json.dumps({
        "query": result.query,
        "overall_consistency": result.consistency.overall_consistency,
        "agreement": result.comparison.agreement_score,
        "recommendation": result.comparison.recommendation,
        "execution_time": result.total_time
    }, indent=2))


if __name__ == "__main__":
    main()

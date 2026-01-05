import json
from core.document import Document
from rag.fid_rag_system import FusionInDecoderRAG


def main():
    documents = [
        Document("d1", "Neural networks learn via backpropagation."),
        Document("d2", "Deep learning uses layered neural networks."),
        Document("d3", "Attention allows models to focus on relevant inputs."),
    ]

    rag = FusionInDecoderRAG()
    rag.index(documents)

    query = "How do neural networks learn?"
    human_answer = "Neural networks learn through backpropagation."

    result = rag.evaluate_full(query, human_answer)

    output = {
        "query": query,
        "overall_consistency": float(result.consistency.overall_consistency),
        "agreement": float(result.comparison.agreement_score),
        "recommendation": result.comparison.recommendation,
        "execution_time": float(result.total_time),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
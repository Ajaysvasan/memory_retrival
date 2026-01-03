import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from core.retrieved_doc import RetrievedDoc
from core.metrics import ConsistencyScore


class ConsistencyEvaluator:
    """
    Consistency evaluation including FiD fusion attention entropy.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[RetrievedDoc],
        fusion_stats: Dict[str, Any]
    ) -> ConsistencyScore:

        if not retrieved_docs:
            return ConsistencyScore(0, 0, 0, 0, 0)

        query_emb = self.model.encode(query)
        doc_embs = [self.model.encode(d.content) for d in retrieved_docs]

        semantic = np.mean([
            np.dot(query_emb, e) /
            (np.linalg.norm(query_emb) * np.linalg.norm(e) + 1e-8)
            for e in doc_embs
        ])

        q_tokens = set(query.lower().split())
        lexical = np.mean([
            len(q_tokens & set(d.content.lower().split())) /
            len(q_tokens | set(d.content.lower().split()))
            for d in retrieved_docs
        ])

        fusion_consistency = 0.0
        if fusion_stats.get("attention_weights"):
            attn = np.array(fusion_stats["attention_weights"])
            entropy = -np.sum(attn * np.log(attn + 1e-8))
            fusion_consistency = 1 - entropy / np.log(len(attn))

        overall = 0.4 * semantic + 0.3 * lexical + 0.3 * fusion_consistency
        variation = np.std(semantic) / (np.mean(semantic) + 1e-8)

        return ConsistencyScore(
            semantic_consistency=float(min(1, semantic)),
            lexical_consistency=float(min(1, lexical)),
            fusion_consistency=float(min(1, fusion_consistency)),
            overall_consistency=float(min(1, overall)),
            variation_coefficient=float(variation)
        )

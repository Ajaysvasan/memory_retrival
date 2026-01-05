"""
================================================================================
AGENTIC RAG (TOOL-ORCHESTRATED / MULTI-STEP RAG) - COMPLETE SYSTEM
================================================================================

All files organized modularly with complete imports.
Directory structure ready to copy and run.

Installation:
pip install sentence-transformers rank-bm25 torch numpy openai langchain
"""

# ============================================================================
# FILE: agentic_rag/core/__init__.py
# ============================================================================

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

@dataclass
class Document:
    """Document structure"""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedDoc:
    """Retrieved document with scores"""
    doc_id: str
    content: str
    retrieval_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComplexityMetrics:
    """Time and Space Complexity"""
    operation: str
    time_complexity: str
    space_complexity: str
    description: str
    empirical_time: float = 0.0

@dataclass
class ConsistencyScore:
    """Consistency metrics"""
    semantic_consistency: float
    lexical_consistency: float
    agent_reasoning_consistency: float
    overall_consistency: float
    variation_coefficient: float

@dataclass
class AccuracyMetrics:
    """Accuracy metrics"""
    bleu_score: float
    rouge_score: float
    similarity_score: float
    f1_score: float
    exact_match: bool

@dataclass
class ComparisonResult:
    """AI vs Human comparison"""
    ai_answer: str
    human_answer: str
    semantic_similarity: float
    lexical_similarity: float
    structural_similarity: float
    length_ratio: float
    agreement_score: float
    recommendation: str

@dataclass
class ToolCall:
    """Tool call representation"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentAction:
    """Agent action in reasoning chain"""
    action_type: str
    action_input: Dict[str, Any]
    reasoning: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    query: str
    retrieved_docs: List[RetrievedDoc]
    generated_answer: str
    consistency: ConsistencyScore
    accuracy: AccuracyMetrics
    comparison: ComparisonResult
    complexity: List[ComplexityMetrics]
    total_time: float
    agent_reasoning_path: List[AgentAction] = field(default_factory=list)
    agent_stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# FILE: agentic_rag/tools/__init__.py
# ============================================================================

from abc import ABC, abstractmethod
from enum import Enum

class ToolType(Enum):
    """Tool types"""
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"

class BaseTool(ABC):
    """Abstract base tool"""
    
    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute tool"""
        pass
    
    def __call__(self, **kwargs) -> Any:
        """Make tool callable"""
        return self.execute(**kwargs)


# ============================================================================
# FILE: agentic_rag/tools/retrieval_tools.py
# ============================================================================

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class SemanticSearchTool(BaseTool):
    """Semantic search using embeddings"""
    
    def __init__(self, documents: List[Document] = None):
        super().__init__(
            name="semantic_search",
            description="Search documents using semantic similarity",
            tool_type=ToolType.RETRIEVAL
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents or []
        self.embeddings = None
        self.index()
    
    def index(self):
        """Index documents"""
        if self.documents:
            contents = [doc.content for doc in self.documents]
            self.embeddings = self.model.encode(contents, convert_to_tensor=True)
    
    def execute(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDoc]:
        """Execute semantic search"""
        if not self.documents or self.embeddings is None:
            return []
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = np.array([
            float(query_emb.dot(emb)) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            for emb in self.embeddings
        ])
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievedDoc(
                doc_id=self.documents[idx].doc_id,
                content=self.documents[idx].content,
                retrieval_score=float(scores[idx]),
                metadata=self.documents[idx].metadata
            ))
        return results


class BM25SearchTool(BaseTool):
    """BM25 lexical search"""
    
    def __init__(self, documents: List[Document] = None):
        super().__init__(
            name="bm25_search",
            description="Search documents using BM25 keyword matching",
            tool_type=ToolType.RETRIEVAL
        )
        self.documents = documents or []
        self.bm25 = None
        self.index()
    
    def index(self):
        """Index documents"""
        if self.documents:
            tokenized = [doc.content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
    
    def execute(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDoc]:
        """Execute BM25 search"""
        if not self.documents or self.bm25 is None:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(RetrievedDoc(
                doc_id=self.documents[idx].doc_id,
                content=self.documents[idx].content,
                retrieval_score=float(scores[idx]),
                metadata=self.documents[idx].metadata
            ))
        return results


class HybridSearchTool(BaseTool):
    """Hybrid search combining semantic and BM25"""
    
    def __init__(self, documents: List[Document] = None, alpha: float = 0.6):
        super().__init__(
            name="hybrid_search",
            description="Search documents using hybrid semantic+BM25 approach",
            tool_type=ToolType.RETRIEVAL
        )
        self.semantic_tool = SemanticSearchTool(documents)
        self.bm25_tool = BM25SearchTool(documents)
        self.alpha = alpha
        self.documents = documents or []
    
    def execute(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDoc]:
        """Execute hybrid search"""
        semantic_results = self.semantic_tool.execute(query, top_k=top_k*2)
        bm25_results = self.bm25_tool.execute(query, top_k=top_k*2)
        
        # Combine scores
        doc_scores = {}
        for doc in semantic_results:
            doc_scores[doc.doc_id] = self.alpha * doc.retrieval_score
        for doc in bm25_results:
            if doc.doc_id in doc_scores:
                doc_scores[doc.doc_id] += (1 - self.alpha) * doc.retrieval_score
            else:
                doc_scores[doc.doc_id] = (1 - self.alpha) * doc.retrieval_score
        
        # Get top results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in sorted_docs:
            doc = next((d for d in self.documents if d.doc_id == doc_id), None)
            if doc:
                results.append(RetrievedDoc(
                    doc_id=doc_id,
                    content=doc.content,
                    retrieval_score=score,
                    metadata=doc.metadata
                ))
        return results


# ============================================================================
# FILE: agentic_rag/tools/reasoning_tools.py
# ============================================================================

class QueryDecompositionTool(BaseTool):
    """Decompose query into sub-questions"""
    
    def __init__(self):
        super().__init__(
            name="decompose_query",
            description="Break complex query into simpler sub-questions",
            tool_type=ToolType.REASONING
        )
    
    def execute(self, query: str, **kwargs) -> List[str]:
        """Decompose query"""
        # Simple rule-based decomposition
        sub_questions = []
        
        if "and" in query.lower():
            parts = query.lower().split("and")
            sub_questions.extend([p.strip().replace("how", "what").capitalize() for p in parts])
        elif "or" in query.lower():
            parts = query.lower().split("or")
            sub_questions.extend([p.strip().capitalize() for p in parts])
        else:
            # Add related questions
            if "how" in query.lower():
                sub_questions.append("What is " + query.replace("How", "").strip() + "?")
            if "why" in query.lower():
                sub_questions.append("What causes " + query.replace("Why", "").strip() + "?")
        
        if not sub_questions:
            sub_questions = [query]
        
        return sub_questions


class QueryReformulationTool(BaseTool):
    """Reformulate query for better retrieval"""
    
    def __init__(self):
        super().__init__(
            name="reformulate_query",
            description="Reformulate query for better search results",
            tool_type=ToolType.REASONING
        )
    
    def execute(self, query: str, feedback: str = "", **kwargs) -> str:
        """Reformulate query"""
        reformulated = query
        
        # Remove question marks for search
        if reformulated.endswith("?"):
            reformulated = reformulated[:-1]
        
        # Add synonyms if feedback provided
        if "unclear" in feedback.lower():
            reformulated += " definition explanation"
        elif "notfound" in feedback.lower():
            reformulated += " related information"
        
        return reformulated


class RelevanceCheckTool(BaseTool):
    """Check if retrieved documents are relevant"""
    
    def __init__(self):
        super().__init__(
            name="check_relevance",
            description="Validate if documents are relevant to query",
            tool_type=ToolType.VALIDATION
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def execute(self, query: str, documents: List[RetrievedDoc], threshold: float = 0.3, **kwargs) -> Dict[str, Any]:
        """Check relevance"""
        query_emb = self.embedding_model.encode(query)
        
        relevant_count = 0
        relevance_scores = []
        
        for doc in documents:
            doc_emb = self.embedding_model.encode(doc.content)
            similarity = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
            relevance_scores.append(similarity)
            if similarity > threshold:
                relevant_count += 1
        
        return {
            "is_relevant": relevant_count / len(documents) > 0.5 if documents else False,
            "relevant_ratio": relevant_count / len(documents) if documents else 0,
            "avg_relevance": np.mean(relevance_scores) if relevance_scores else 0,
            "relevance_scores": relevance_scores
        }


# ============================================================================
# FILE: agentic_rag/tools/synthesis_tools.py
# ============================================================================

class AnswerSynthesisTool(BaseTool):
    """Synthesize answer from documents"""
    
    def __init__(self):
        super().__init__(
            name="synthesize_answer",
            description="Generate final answer from retrieved documents",
            tool_type=ToolType.SYNTHESIS
        )
    
    def execute(self, query: str, documents: List[RetrievedDoc], **kwargs) -> str:
        """Synthesize answer"""
        if not documents:
            return "No relevant documents found to answer the query."
        
        # Combine documents
        combined = " ".join([doc.content for doc in documents[:3]])
        sentences = combined.split(".")
        
        # Extract relevant sentences
        query_tokens = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_tokens = set(sentence.lower().split())
            overlap = len(query_tokens & sentence_tokens)
            if overlap > 1:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = ". ".join(relevant_sentences[:5])
            if answer:
                answer += "."
        else:
            answer = combined[:300] + "..."
        
        return answer


class AnswerValidationTool(BaseTool):
    """Validate synthesized answer"""
    
    def __init__(self):
        super().__init__(
            name="validate_answer",
            description="Check if answer is complete and accurate",
            tool_type=ToolType.VALIDATION
        )
    
    def execute(self, answer: str, query: str, **kwargs) -> Dict[str, Any]:
        """Validate answer"""
        answer_tokens = set(answer.lower().split())
        query_tokens = set(query.lower().split())
        
        coverage = len(answer_tokens & query_tokens) / len(query_tokens) if query_tokens else 0
        length_ok = len(answer) > 50
        completeness = coverage > 0.3 and length_ok
        
        return {
            "is_valid": completeness,
            "coverage": coverage,
            "length_ok": length_ok,
            "confidence": coverage * 0.7 + (0.3 if length_ok else 0)
        }


# ============================================================================
# FILE: agentic_rag/agent/__init__.py
# ============================================================================

from enum import Enum
from typing import Callable

class AgentState(Enum):
    """Agent states"""
    INIT = "init"
    DECOMPOSING = "decomposing"
    RETRIEVING = "retrieving"
    VALIDATING = "validating"
    REASONING = "reasoning"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"


class ToolRegistry:
    """Registry of available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register tool"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())


# ============================================================================
# FILE: agentic_rag/agent/orchestrator.py
# ============================================================================

import time
from collections import defaultdict

class AgentOrchestrator:
    """Orchestrates multi-step RAG with tools"""
    
    def __init__(self, max_iterations: int = 5):
        self.registry = ToolRegistry()
        self.max_iterations = max_iterations
        self.reasoning_path: List[AgentAction] = []
        self.state = AgentState.INIT
        self.iteration_count = 0
        self.tool_usage_stats = defaultdict(int)
    
    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        self.registry.register(tool)
    
    def get_reasoning_path(self) -> List[AgentAction]:
        """Get agent reasoning path"""
        return self.reasoning_path
    
    def execute_tool(self, tool_name: str, **kwargs) -> Tuple[Any, ToolCall]:
        """Execute a tool"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        start_time = time.time()
        try:
            result = tool.execute(**kwargs)
            elapsed = time.time() - start_time
            
            tool_call = ToolCall(
                tool_name=tool_name,
                tool_input=kwargs,
                tool_output=result,
                timestamp=time.time()
            )
            
            self.tool_usage_stats[tool_name] += 1
            return result, tool_call
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}")
    
    def plan(self, query: str) -> List[str]:
        """Plan execution steps"""
        self.state = AgentState.DECOMPOSING
        
        # Step 1: Decompose query
        decompose_tool = self.registry.get_tool("decompose_query")
        if decompose_tool:
            sub_queries, tool_call = self.execute_tool("decompose_query", query=query)
            action = AgentAction(
                action_type="DECOMPOSE",
                action_input={"query": query},
                reasoning=f"Decomposed query into {len(sub_queries)} sub-questions",
                tool_calls=[tool_call]
            )
            self.reasoning_path.append(action)
            return sub_queries
        return [query]
    
    def retrieve(self, queries: List[str], retrieval_method: str = "hybrid_search", top_k: int = 5) -> List[RetrievedDoc]:
        """Retrieve documents for queries"""
        self.state = AgentState.RETRIEVING
        all_docs = {}
        
        for query in queries:
            result, tool_call = self.execute_tool(retrieval_method, query=query, top_k=top_k)
            
            for doc in result:
                if doc.doc_id not in all_docs:
                    all_docs[doc.doc_id] = doc
        
        action = AgentAction(
            action_type="RETRIEVE",
            action_input={"queries": queries, "method": retrieval_method},
            reasoning=f"Retrieved {len(all_docs)} unique documents",
            tool_calls=[]
        )
        self.reasoning_path.append(action)
        
        return list(all_docs.values())
    
    def validate(self, query: str, documents: List[RetrievedDoc]) -> Dict[str, Any]:
        """Validate retrieved documents"""
        self.state = AgentState.VALIDATING
        
        relevance_result, tool_call = self.execute_tool(
            "check_relevance",
            query=query,
            documents=documents,
            threshold=0.3
        )
        
        action = AgentAction(
            action_type="VALIDATE",
            action_input={"query": query, "doc_count": len(documents)},
            reasoning=f"Relevance check: {relevance_result['relevant_ratio']:.2%} documents relevant",
            tool_calls=[tool_call]
        )
        self.reasoning_path.append(action)
        
        return relevance_result
    
    def synthesize(self, query: str, documents: List[RetrievedDoc]) -> str:
        """Synthesize answer"""
        self.state = AgentState.SYNTHESIZING
        
        answer, tool_call = self.execute_tool(
            "synthesize_answer",
            query=query,
            documents=documents
        )
        
        # Validate answer
        validation, val_tool = self.execute_tool(
            "validate_answer",
            answer=answer,
            query=query
        )
        
        action = AgentAction(
            action_type="SYNTHESIZE",
            action_input={"query": query, "doc_count": len(documents)},
            reasoning=f"Generated answer with confidence {validation['confidence']:.2%}",
            tool_calls=[tool_call, val_tool]
        )
        self.reasoning_path.append(action)
        
        return answer
    
    def execute(self, query: str, retrieval_method: str = "hybrid_search") -> Tuple[str, List[RetrievedDoc], List[AgentAction]]:
        """Execute full agentic RAG pipeline"""
        self.state = AgentState.INIT
        self.reasoning_path = []
        self.iteration_count = 0
        
        # Step 1: Plan (decompose query)
        sub_queries = self.plan(query)
        
        # Step 2: Retrieve
        documents = self.retrieve(sub_queries, retrieval_method)
        
        # Step 3: Validate
        relevance_check = self.validate(query, documents)
        
        # Step 4: Synthesize
        answer = self.synthesize(query, documents)
        
        self.state = AgentState.COMPLETE
        
        return answer, documents, self.reasoning_path


# ============================================================================
# FILE: agentic_rag/evaluators/__init__.py
# ============================================================================

from collections import Counter
from difflib import SequenceMatcher

class ConsistencyEvaluator:
    """Evaluate consistency"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def evaluate(self, query: str, documents: List[RetrievedDoc], reasoning_path: List[AgentAction]) -> ConsistencyScore:
        """Evaluate consistency"""
        if not documents:
            return ConsistencyScore(0, 0, 0, 0, 0)
        
        # Semantic consistency
        query_emb = self.embedding_model.encode(query)
        doc_embs = [self.embedding_model.encode(doc.content) for doc in documents]
        
        semantic_sims = [
            float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
            for emb in doc_embs
        ]
        semantic_consistency = float(np.mean(semantic_sims))
        
        # Lexical consistency
        query_tokens = set(query.lower().split())
        lexical_sims = []
        for doc in documents:
            doc_tokens = set(doc.content.lower().split())
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            jaccard = intersection / union if union > 0 else 0
            lexical_sims.append(jaccard)
        
        lexical_consistency = float(np.mean(lexical_sims))
        
        # Agent reasoning consistency
        agent_consistency = len(reasoning_path) / 5.0  # 5 steps ideal
        agent_consistency = min(1.0, agent_consistency)
        
        overall_consistency = 0.4 * semantic_consistency + 0.3 * lexical_consistency + 0.3 * agent_consistency
        variation_coefficient = float(np.std(semantic_sims) / (np.mean(semantic_sims) + 1e-8))
        
        return ConsistencyScore(
            semantic_consistency=min(1.0, semantic_consistency),
            lexical_consistency=min(1.0, lexical_consistency),
            agent_reasoning_consistency=agent_consistency,
            overall_consistency=min(1.0, overall_consistency),
            variation_coefficient=variation_coefficient
        )


class AccuracyEvaluator:
    """Evaluate accuracy"""
    
    @staticmethod
    def _bleu_score(reference: str, candidate: str) -> float:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if not cand_tokens:
            return 0.0
        overlap = sum((Counter(ref_tokens) & Counter(cand_tokens)).values())
        precision = overlap / len(cand_tokens)
        if not ref_tokens:
            return 0.0
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1)
    
    @staticmethod
    def _rouge_score(reference: str, candidate: str) -> float:
        matcher = SequenceMatcher(None, reference, candidate)
        return min(1.0, matcher.ratio())
    
    @staticmethod
    def _similarity_score(str1: str, str2: str) -> float:
        matcher = SequenceMatcher(None, str1.lower(), str2.lower())
        return min(1.0, matcher.ratio())
    
    @staticmethod
    def _f1_score(reference: str, candidate: str) -> float:
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        if not cand_tokens or not ref_tokens:
            return 0.0
        intersection = len(ref_tokens & cand_tokens)
        precision = intersection / len(cand_tokens)
        recall = intersection / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return min(1.0, f1)
    
    def evaluate(self, ai_answer: str, reference_answer: str) -> AccuracyMetrics:
        """Evaluate accuracy"""
        return AccuracyMetrics(
            bleu_score=self._bleu_score(reference_answer, ai_answer),
            rouge_score=self._rouge_score(reference_answer, ai_answer),
            similarity_score=self._similarity_score(ai_answer, reference_answer),
            f1_score=self._f1_score(reference_answer, ai_answer),
            exact_match=ai_answer.lower().strip() == reference_answer.lower().strip()
        )


class ComparisonEvaluator:
    """Compare AI vs Human"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def evaluate(self, ai_answer: str, human_answer: str) -> ComparisonResult:
        """Compare answers"""
        ai_emb = self.embedding_model.encode(ai_answer)
        human_emb = self.embedding_model.encode(human_answer)
        semantic_sim = float(np.dot(ai_emb, human_emb) / (np.linalg.norm(ai_emb) * np.linalg.norm(human_emb) + 1e-8))
        
        ai_tokens = set(ai_answer.lower().split())
        human_tokens = set(human_answer.lower().split())
        intersection = len(ai_tokens & human_tokens)
        union = len(ai_tokens | human_tokens)
        lexical_sim = intersection / union if union > 0 else 0
        
        matcher = SequenceMatcher(None, ai_answer, human_answer)
        structural_sim = matcher.ratio()
        
        ai_len = len(ai_answer.split())
        human_len = len(human_answer.split())
        length_ratio = ai_len / human_len if human_len > 0 else 1.0
        
        agreement = 0.5 * semantic_sim + 0.3 * lexical_sim + 0.2 * structural_sim
        
        if agreement > 0.85:
            recommendation = "ACCEPT - Excellent agreement"
        elif agreement > 0.70:
            recommendation = "REVIEW - Good agreement"
        elif agreement > 0.50:
            recommendation = "REVIEW - Moderate agreement"
        else:
            recommendation = "REJECT - Significant differences"
        
        return ComparisonResult(
            ai_answer=ai_answer,
            human_answer=human_answer,
            semantic_similarity=min(1.0, semantic_sim),
            lexical_similarity=min(1.0, lexical_sim),
            structural_similarity=min(1.0, structural_sim),
            length_ratio=min(2.0, length_ratio),
            agreement_score=min(1.0, agreement),
            recommendation=recommendation
        )


class ComplexityAnalyzer:
    """Analyze complexity"""
    
    @staticmethod
    def get_complexity_analysis(num_tools: int, num_docs: int, steps: int) -> List[ComplexityMetrics]:
        """Get complexity metrics"""
        return [
            ComplexityMetrics(
                operation="Tool Registration",
                time_complexity="O(T) where T=num_tools",
                space_complexity="O(T)",
                description=f"Register {num_tools} tools in registry"
            ),
            ComplexityMetrics(
                operation="Query Decomposition",
                time_complexity="O(L) where L=query_length",
                space_complexity="O(Q) where Q=num_sub_questions",
                description="Break query into sub-questions"
            ),
            ComplexityMetrics(
                operation="Multi-step Retrieval",
                time_complexity="O(Q * (n*d + Retrieval)) where Q=sub_queries, n=docs",
                space_complexity="O(Q * R) where R=top_k retrieved",
                description=f"Retrieve for {steps} decomposed queries"
            ),
            ComplexityMetrics(
                operation="Relevance Validation",
                time_complexity="O(D * d) where D=retrieved_docs, d=embedding_dim",
                space_complexity="O(D * d)",
                description="Validate each retrieved document"
            ),
            ComplexityMetrics(
                operation="Agent Reasoning",
                time_complexity="O(S * T) where S=steps, T=avg_tool_time",
                space_complexity="O(S * Path_length)",
                description="Execute agent reasoning steps"
            ),
            ComplexityMetrics(
                operation="Answer Synthesis",
                time_complexity="O(D * L) where D=docs, L=doc_length",
                space_complexity="O(A) where A=answer_length",
                description="Generate final answer from documents"
            ),
            ComplexityMetrics(
                operation="Full Agentic RAG",
                time_complexity="O(Q * n*d + S*T) where Q=decomposed, S=steps",
                space_complexity="O(n*d + S*Path_length)",
                description="Complete agentic RAG pipeline"
            ),
        ]


# ============================================================================
# FILE: agentic_rag/system/__init__.py
# ============================================================================

class AgenticRAGSystem:
    """Complete Agentic RAG System"""
    
    def __init__(self, max_iterations: int = 5, retrieval_method: str = "hybrid_search"):
        self.orchestrator = AgentOrchestrator(max_iterations=max_iterations)
        self.consistency_evaluator = ConsistencyEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.comparison_evaluator = ComparisonEvaluator()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.retrieval_method = retrieval_method
        self.documents = []
        self.n_docs = 0
        
        print(f"[AgenticRAG] Initialized with max_iterations={max_iterations}")
    
    def setup(self, documents: List[Document]):
        """Setup with documents"""
        self.documents = documents
        self.n_docs = len(documents)
        
        # Register tools
        semantic_tool = SemanticSearchTool(documents)
        bm25_tool = BM25SearchTool(documents)
        hybrid_tool = HybridSearchTool(documents)
        
        self.orchestrator.register_tool(semantic_tool)
        self.orchestrator.register_tool(bm25_tool)
        self.orchestrator.register_tool(hybrid_tool)
        
        # Register reasoning tools
        self.orchestrator.register_tool(QueryDecompositionTool())
        self.orchestrator.register_tool(QueryReformulationTool())
        self.orchestrator.register_tool(RelevanceCheckTool())
        self.orchestrator.register_tool(AnswerSynthesisTool())
        self.orchestrator.register_tool(AnswerValidationTool())
        
        print(f"[AgenticRAG] Indexed {self.n_docs} documents")
    
    def evaluate_full(self, query: str, human_answer: str) -> EvaluationResult:
        """Complete evaluation"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[AgenticRAG] Processing Query: {query[:50]}...")
        print(f"{'='*70}\n")
        
        # Execute agentic pipeline
        print("[Stage 1] Planning and decomposing query...")
        answer, retrieved_docs, reasoning_path = self.orchestrator.execute(query, self.retrieval_method)
        print(f"  ✓ Generated {len(reasoning_path)} reasoning steps\n")
        
        # Evaluate consistency
        print("[Stage 2] Evaluating consistency...")
        consistency = self.consistency_evaluator.evaluate(query, retrieved_docs, reasoning_path)
        print(f"  ✓ Overall Consistency: {consistency.overall_consistency:.4f}\n")
        
        # Evaluate accuracy
        print("[Stage 3] Evaluating accuracy...")
        accuracy = self.accuracy_evaluator.evaluate(answer, human_answer)
        print(f"  ✓ BLEU: {accuracy.bleu_score:.4f}, F1: {accuracy.f1_score:.4f}\n")
        
        # Compare with human
        print("[Stage 4] Comparing with human answer...")
        comparison = self.comparison_evaluator.evaluate(answer, human_answer)
        print(f"  ✓ Agreement: {comparison.agreement_score:.4f}\n")
        
        # Analyze complexity
        print("[Stage 5] Analyzing complexity...")
        complexity = self.complexity_analyzer.get_complexity_analysis(
            len(self.orchestrator.registry.list_tools()),
            self.n_docs,
            len(reasoning_path)
        )
        print(f"  ✓ Generated {len(complexity)} metrics\n")
        
        total_time = time.time() - start_time
        
        # Prepare agent stats
        agent_stats = {
            "total_steps": len(reasoning_path),
            "tool_usage": dict(self.orchestrator.tool_usage_stats),
            "reasoning_depth": len(reasoning_path),
            "tools_available": len(self.orchestrator.registry.list_tools())
        }
        
        return EvaluationResult(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_answer=answer,
            consistency=consistency,
            accuracy=accuracy,
            comparison=comparison,
            complexity=complexity,
            total_time=total_time,
            agent_reasoning_path=reasoning_path,
            agent_stats=agent_stats
        )


# ============================================================================
# FILE: main.py
# ============================================================================

import json

def create_sample_documents() -> List[Document]:
    """Create sample documents"""
    return [
        Document(doc_id="doc1", content="Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data without being explicitly programmed. ML algorithms identify patterns in data.", metadata={"source": "AI Basics"}),
        Document(doc_id="doc2", content="Neural networks are computing systems inspired by biological neural networks that constitute animal brains. They consist of interconnected nodes that process information.", metadata={"source": "NN"}),
        Document(doc_id="doc3", content="Deep learning uses multiple layers of neural networks to learn hierarchical representations. It's particularly effective for image and language processing tasks.", metadata={"source": "DL"}),
        Document(doc_id="doc4", content="Backpropagation is the fundamental algorithm for training neural networks. It computes gradients of the loss function with respect to weights using the chain rule.", metadata={"source": "Training"}),
        Document(doc_id="doc5", content="Gradient descent is an optimization algorithm that iteratively moves parameters in the direction of negative gradient to minimize the loss function during training.", metadata={"source": "Optimization"}),
        Document(doc_id="doc6", content="Transformers use self-attention mechanisms to process sequential data. Models like BERT and GPT have achieved state-of-the-art results in natural language processing.", metadata={"source": "Transformers"}),
        Document(doc_id="doc7", content="Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. They use convolutional layers with shared weights to detect features.", metadata={"source": "CNN"}),
        Document(doc_id="doc8", content="Recurrent Neural Networks (RNNs) process sequential data by maintaining hidden states. LSTMs and GRUs are variants that handle long-term dependencies better.", metadata={"source": "RNN"}),
        Document(doc_id="doc9", content="Transfer learning involves taking a model trained on one task and adapting it for another task. This approach reduces training time and improves performance with limited data.", metadata={"source": "Transfer"}),
        Document(doc_id="doc10", content="Regularization techniques like dropout and L1/L2 regularization prevent overfitting by constraining model complexity. They improve generalization to unseen data.", metadata={"source": "Regularization"}),
    ]


def format_results(result: EvaluationResult) -> Dict:
    """Format results"""
    return {
        "system_type": "Agentic RAG (Tool-Orchestrated)",
        "query": result.query,
        "generated_ai_answer": result.generated_answer,
        "human_reference_answer": result.comparison.human_answer,
        
        "consistency_metrics": {
            "semantic_consistency": round(result.consistency.semantic_consistency, 4),
            "lexical_consistency": round(result.consistency.lexical_consistency, 4),
            "agent_reasoning_consistency": round(result.consistency.agent_reasoning_consistency, 4),
            "overall_consistency": round(result.consistency.overall_consistency, 4),
            "variation_coefficient": round(result.consistency.variation_coefficient, 4),
        },
        
        "accuracy_metrics": {
            "bleu_score": round(result.accuracy.bleu_score, 4),
            "rouge_score": round(result.accuracy.rouge_score, 4),
            "similarity_score": round(result.accuracy.similarity_score, 4),
            "f1_score": round(result.accuracy.f1_score, 4),
            "exact_match": result.accuracy.exact_match,
        },
        
        "ai_vs_human_comparison": {
            "semantic_similarity": round(result.comparison.semantic_similarity, 4),
            "lexical_similarity": round(result.comparison.lexical_similarity, 4),
            "structural_similarity": round(result.comparison.structural_similarity, 4),
            "length_ratio": round(result.comparison.length_ratio, 4),
            "agreement_score": round(result.comparison.agreement_score, 4),
            "recommendation": result.comparison.recommendation,
        },
        
        "agent_reasoning_path": [
            {
                "step": i+1,
                "action_type": action.action_type,
                "reasoning": action.reasoning,
                "tools_used": len(action.tool_calls),
            }
            for i, action in enumerate(result.agent_reasoning_path)
        ],
        
        "agent_statistics": {
            "total_reasoning_steps": result.agent_stats.get("total_steps"),
            "reasoning_depth": result.agent_stats.get("reasoning_depth"),
            "tools_available": result.agent_stats.get("tools_available"),
            "tool_usage": result.agent_stats.get("tool_usage"),
        },
        
        "retrieved_documents": [
            {
                "doc_id": doc.doc_id,
                "content_preview": doc.content[:80] + "...",
                "retrieval_score": round(doc.retrieval_score, 4),
                "metadata": doc.metadata
            }
            for doc in result.retrieved_docs[:5]
        ],
        
        "complexity_analysis": [
            {
                "operation": c.operation,
                "time_complexity": c.time_complexity,
                "space_complexity": c.space_complexity,
            }
            for c in result.complexity[:5]
        ],
        
        "performance": {
            "total_execution_time_seconds": round(result.total_time, 4),
            "documents_indexed": len([d for d in result.retrieved_docs]),
        }
    }


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("AGENTIC RAG (TOOL-ORCHESTRATED / MULTI-STEP RAG)")
    print("="*70 + "\n")
    
    # Initialize
    print("[Init] Creating Agentic RAG system...")
    agentic_rag = AgenticRAGSystem(max_iterations=5, retrieval_method="hybrid_search")
    
    # Load documents
    print("[Data] Loading documents...")
    documents = create_sample_documents()
    print(f"  ✓ Loaded {len(documents)} documents\n")
    
    # Setup
    print("[Setup] Configuring tools and registering documents...")
    agentic_rag.setup(documents)
    
    # Query
    query = "How do neural networks learn from data through training?"
    human_answer = "Neural networks learn through backpropagation, which computes gradients and updates weights using gradient descent. This process iteratively minimizes the loss function, allowing the network to learn patterns from data."
    
    # Evaluate
    print(f"\n[Eval] Running agentic RAG pipeline...")
    result = agentic_rag.evaluate_full(query, human_answer)
    
    # Display
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    
    output = format_results(result)
    print(json.dumps(output, indent=2))
    
    print(f"\n{'='*70}")
    print(f"Total Execution Time: {result.total_time:.4f} seconds")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
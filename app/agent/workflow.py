from typing import Any, Dict

from app.agent.planner import classify_query, choose_strategy
from app.agent.tools import (
    run_retrieval,
    run_reranking,
    check_evidence_sufficiency,
    generate_grounded_answer,
    run_citation_validation,
)
from app.retrieval.hybrid import HybridRetriever
from app.reranking.cross_encoder import CrossEncoderReranker
from app.generation.generator import GroundedAnswerGenerator

# Special refusal message when evidence is insufficient
REFUSAL_MESSAGE = "I could not find enough support in the indexed documents."

def execute_agent(
    query: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    generator: GroundedAnswerGenerator,
) -> Dict[str, Any]:
    """
    End-to-end execution of the simple rule-based agent.
    """
    # 1. Inspect & classify
    query_type = classify_query(query)

    # 2. Choose strategy
    strategy = choose_strategy(query_type)

    # 3. First pass retrieval & reranking
    candidates = run_retrieval(query, retriever, strategy)
    reranked = run_reranking(query, candidates, reranker, strategy)

    # 4. Check evidence sufficiency
    # Assuming threshold > 0 as some cross-encoders can output negative scores. 
    # For ms-marco-MiniLM-L-6-v2, scores can be unnormalized logits. 
    # We will use 0.0 as a baseline generic threshold for logits, or we can use -2.0 to be safe.
    # Let's use a conservative 0.0 or let the tool use its default 0.5.
    is_sufficient = check_evidence_sufficiency(reranked, threshold=0.5)

    if not is_sufficient:
        # 5. Try a fallback retrieval pass with relaxed parameters
        #    E.g. increase top_k to search broader
        fallback_strategy = strategy.copy()
        fallback_strategy["bm25_top_k"] += 20
        fallback_strategy["vector_top_k"] += 20
        
        fallback_candidates = run_retrieval(query, retriever, fallback_strategy)
        fallback_reranked = run_reranking(query, fallback_candidates, reranker, fallback_strategy)
        
        # 6. Check again
        is_sufficient = check_evidence_sufficiency(fallback_reranked, threshold=0.0)
        reranked = fallback_reranked

    if not is_sufficient:
        # If still weak, return refusal message mimicking a dictionary result
        return {
            "query": query,
            "query_type": query_type,
            "strategy": strategy,
            "is_sufficient": False,
            "answer": REFUSAL_MESSAGE,
            "citation_validation": None,
            "citations": []
        }

    # 7. Generate answer
    gen_result = generate_grounded_answer(query, reranked, generator)

    # 8. Validate citations
    validation_result = run_citation_validation(gen_result.answer, reranked)

    # 9. Format structured final result
    return {
        "query": query,
        "query_type": query_type,
        "strategy": strategy,
        "is_sufficient": True,
        "answer": gen_result.answer,
        "citation_validation": validation_result,
        "citations": [c.chunk_id for c in validation_result.citations] if validation_result else []
    }

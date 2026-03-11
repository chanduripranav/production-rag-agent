from typing import Dict, Any, List

from app.retrieval.hybrid import HybridRetriever, HybridResult
from app.reranking.cross_encoder import CrossEncoderReranker, RerankedResult
from app.generation.generator import GroundedAnswerGenerator, GenerationResult
from app.citations.validator import validate_citations, CitationValidationResult

def run_retrieval(query: str, retriever: HybridRetriever, strategy_params: Dict[str, Any]) -> List[HybridResult]:
    """
    Wrap the HybridRetriever search, passing the selected strategy parameters.
    """
    return retriever.search(
        query=query,
        top_k=strategy_params["top_k"],
        bm25_top_k=strategy_params.get("bm25_top_k", 25),
        vector_top_k=strategy_params.get("vector_top_k", 25),
        rrf_k=strategy_params.get("rrf_k", 60)
    )

def run_reranking(query: str, candidates: List[HybridResult], reranker: CrossEncoderReranker, strategy_params: Dict[str, Any]) -> List[RerankedResult]:
    """
    Wrap the CrossEncoderReranker.
    """
    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=strategy_params["top_k"]
    )

def check_evidence_sufficiency(reranked_results: List[RerankedResult], threshold: float = 0.5) -> bool:
    """
    Check whether the top piece of evidence has a high enough score.
    Returns True if valid evidence exists.
    """
    if not reranked_results:
        return False
    
    # Using a simple heuristic on the top rerank score
    top_score = reranked_results[0].rerank_score
    return top_score >= threshold

def generate_grounded_answer(query: str, context: List[RerankedResult], generator: GroundedAnswerGenerator) -> GenerationResult:
    """
    Wrap the generator to build an answer from the query and context.
    """
    return generator.generate(question=query, chunks=context)

def run_citation_validation(answer_text: str, context_chunks: List[RerankedResult]) -> CitationValidationResult:
    """
    Wrap the citation validation logic.
    """
    return validate_citations(answer_text=answer_text, context_chunks=context_chunks)

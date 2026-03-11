from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import List

from app.api.schemas import CitationOut, ChunkOut, QueryResponse
from app.citations.validator import validate_citations
from app.generation.generator import GroundedAnswerGenerator, MockLLMClient
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.hybrid import HybridRetriever


def repo_root() -> Path:
    """
    Compute the repo root directory.

    service.py -> app/api/service.py
    parents[0] = api/, parents[1] = app/, parents[2] = repo root
    """
    return Path(__file__).resolve().parents[2]


def _preview(text: str, *, max_chars: int = 200) -> str:
    """
    Short preview used in API responses.
    """
    cleaned = (text or "").replace("\n", " ").strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


@lru_cache(maxsize=1)
def _hybrid_retriever() -> HybridRetriever:
    """
    Cache the retriever so we don't rebuild BM25 / reload FAISS on every request.
    """
    root = repo_root()
    return HybridRetriever(
        chunks_path=root / "data" / "processed" / "chunks.json",
        vector_index_path=root / "data" / "processed" / "vector.index",
        vector_meta_path=root / "data" / "processed" / "vector_meta.json",
    )


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoderReranker:
    """
    Cache the reranker so the cross-encoder model is loaded once.
    """
    return CrossEncoderReranker()


@lru_cache(maxsize=1)
def _generator() -> GroundedAnswerGenerator:
    """
    Default generator uses the mock LLM for now (safe and fast).

    Later you can swap this for a real LLM client.
    """
    return GroundedAnswerGenerator(llm=MockLLMClient())


class QueryService:
    """
    Orchestrates the end-to-end query pipeline for the API.

    Pipeline:
    1) hybrid retrieval
    2) reranking
    3) grounded generation
    4) citation validation
    """

    def run(self, question: str) -> QueryResponse:
        hybrid = _hybrid_retriever()
        reranker = _reranker()
        generator = _generator()

        # 1) Retrieve candidates
        candidates = hybrid.search(question, top_k=25)

        # 2) Rerank (use top 5 as our final context for generation/validation)
        reranked = reranker.rerank(question, candidates, top_k=5)

        # 3) Generate answer
        gen = generator.generate(question, reranked)

        # 4) Validate citations against the reranked context
        validation = validate_citations(gen.answer, reranked)

        citations_out: List[CitationOut] = [
            CitationOut(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                page_number=c.page_number,
                text_preview=c.text_preview,
            )
            for c in validation.citations
        ]

        top_chunks_out: List[ChunkOut] = [
            ChunkOut(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                page_number=c.page_number,
                text_preview=_preview(c.text),
            )
            for c in reranked
        ]

        return QueryResponse(
            question=question,
            answer=gen.answer,
            citation_validation_passed=validation.passed,
            citations=citations_out,
            top_chunks=top_chunks_out,
        )


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """
    Dependency provider for FastAPI (easy to override in tests).
    """
    return QueryService()


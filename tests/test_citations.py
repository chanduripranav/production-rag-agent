from __future__ import annotations

from app.citations.validator import validate_citations
from app.reranking.cross_encoder import RerankedResult


def test_valid_citations_pass() -> None:
    context = [
        RerankedResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="hello", rerank_score=1.0),
        RerankedResult(chunk_id="c2", source_file="b.pdf", page_number=2, text="world", rerank_score=0.9),
    ]
    answer = "This is supported by the docs [c1] and also [c2]."
    result = validate_citations(answer, context)
    assert result.passed is True
    assert result.invalid_citations == []
    assert [c.chunk_id for c in result.citations] == ["c1", "c2"]


def test_fake_citation_fails() -> None:
    context = [RerankedResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="hello", rerank_score=1.0)]
    answer = "This is not real [does-not-exist]."
    result = validate_citations(answer, context)
    assert result.passed is False
    assert result.invalid_citations == ["does-not-exist"]
    assert result.citations == []


def test_duplicate_citations_are_deduped_cleanly() -> None:
    context = [RerankedResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="hello", rerank_score=1.0)]
    answer = "Repeat [c1] again [c1]."
    result = validate_citations(answer, context)
    assert result.passed is True
    assert [c.chunk_id for c in result.citations] == ["c1"]


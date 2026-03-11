from __future__ import annotations

from app.generation.generator import (
    REFUSAL_MESSAGE,
    GroundedAnswerGenerator,
    MockLLMClient,
    extract_citations,
)
from app.generation.prompt import build_grounded_prompt
from app.reranking.cross_encoder import RerankedResult


def test_prompt_contains_question_and_chunks() -> None:
    chunks = [
        RerankedResult(
            chunk_id="c1",
            source_file="a.pdf",
            page_number=1,
            text="Fourier transform is mentioned here.",
            rerank_score=1.0,
        )
    ]
    parts = build_grounded_prompt("What is Fourier transform?", chunks)
    prompt = parts.render()

    assert "QUESTION:" in prompt
    assert "What is Fourier transform?" in prompt
    assert "CONTEXT:" in prompt
    assert "[chunk_id=c1]" in prompt
    assert "Answer ONLY using the provided CONTEXT" in prompt
    assert "citations in the form [chunk_id]" in prompt


def test_mock_generator_includes_citations() -> None:
    chunks = [
        RerankedResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="Fourier transform ...", rerank_score=1.0),
        RerankedResult(chunk_id="c2", source_file="b.pdf", page_number=2, text="More context ...", rerank_score=0.9),
    ]

    gen = GroundedAnswerGenerator(llm=MockLLMClient())
    result = gen.generate("What is Fourier transform?", chunks)

    assert result.answer != REFUSAL_MESSAGE
    assert "[c1]" in result.answer or "[c2]" in result.answer
    assert len(result.citations) >= 1


def test_refusal_when_no_context() -> None:
    gen = GroundedAnswerGenerator(llm=MockLLMClient())
    result = gen.generate("Any question", [])
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


def test_extract_citations_unique_and_ordered() -> None:
    text = "A [c2] B [c1] C [c2]"
    assert extract_citations(text) == ["c2", "c1"]


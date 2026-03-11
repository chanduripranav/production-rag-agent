from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Protocol, Sequence

from app.generation.prompt import PromptParts, build_grounded_prompt
from app.reranking.cross_encoder import RerankedResult


REFUSAL_MESSAGE = "I could not find enough support in the indexed documents."


class LLMClient(Protocol):
    """
    Pluggable LLM client interface.

    Later you can implement a real client (OpenAI, Ollama, LM Studio, etc.)
    by providing a class with a `generate(prompt: str) -> str` method.
    """

    def generate(self, prompt: str) -> str:  # pragma: no cover
        ...


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations: List[str]


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_citations(text: str) -> List[str]:
    """
    Extract citations of the form [chunk_id] from the model output.
    """
    found = _CITATION_RE.findall(text or "")
    # Keep order while removing duplicates.
    seen = set()
    out: List[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class MockLLMClient:
    """
    A simple mock LLM that produces a grounded-looking answer.

    Important:
    - It does NOT invent facts outside the provided chunks.
    - It uses short snippets from the top chunks.
    - It always adds citations like [chunk_id].
    """

    def __init__(self, *, max_snippet_chars: int = 240):
        self._max_snippet_chars = max_snippet_chars

    def generate(self, prompt: str) -> str:
        # This mock does not parse the entire prompt perfectly; instead it relies on the
        # pipeline passing chunks separately (see GroundedAnswerGenerator below).
        return prompt

    def generate_from_chunks(self, question: str, chunks: Sequence[RerankedResult]) -> str:
        if not chunks:
            return REFUSAL_MESSAGE

        # Build an answer that quotes/uses the chunk text itself.
        parts: List[str] = []
        parts.append(f"Based on the provided documents, here is what I found:")

        used = 0
        for c in chunks:
            snippet = (c.text or "").strip().replace("\n", " ")
            if not snippet:
                continue
            if len(snippet) > self._max_snippet_chars:
                snippet = snippet[: self._max_snippet_chars].rstrip() + "..."
            parts.append(f"- {snippet} [{c.chunk_id}]")
            used += 1
            if used >= 2:
                break

        if used == 0:
            return REFUSAL_MESSAGE

        # End with a short direct response line using citations.
        parts.append(f"\nIf you want, ask a more specific question and I will cite the exact chunk IDs.")
        return "\n".join(parts).strip()


class LocalLLMStubClient:
    """
    A "real generator option" placeholder.

    This is where you'd connect to a local LLM server (Ollama, LM Studio, etc.).
    For now it raises an error with a clear message so beginners know what to do next.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "LocalLLMStubClient is a stub. Plug in a real LLM client by implementing "
            "`generate(prompt: str) -> str` and passing it to GroundedAnswerGenerator."
        )


class GroundedAnswerGenerator:
    """
    Builds a grounded prompt and calls an LLM client to generate the final answer.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def generate(self, question: str, chunks: Sequence[RerankedResult]) -> GenerationResult:
        # Refuse early if there is no usable context.
        if not chunks or all((not (c.text or "").strip()) for c in chunks):
            return GenerationResult(answer=REFUSAL_MESSAGE, citations=[])

        prompt_parts: PromptParts = build_grounded_prompt(question, chunks)
        prompt = prompt_parts.render()

        # Special case: if the caller passed our MockLLMClient, use its helper
        # to ensure the answer is based on the chunk texts.
        if isinstance(self._llm, MockLLMClient):
            answer = self._llm.generate_from_chunks(question, chunks)
        else:
            answer = self._llm.generate(prompt)

        # If the model refused, keep it as-is.
        if answer.strip() == REFUSAL_MESSAGE:
            return GenerationResult(answer=REFUSAL_MESSAGE, citations=[])

        citations = extract_citations(answer)
        return GenerationResult(answer=answer.strip(), citations=citations)


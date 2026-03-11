from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.generation.generator import extract_citations
from app.reranking.cross_encoder import RerankedResult


@dataclass(frozen=True)
class CitationMeta:
    """
    Structured info about a validated citation.
    """

    chunk_id: str
    source_file: str
    page_number: int
    text_preview: str


@dataclass(frozen=True)
class CitationValidationResult:
    """
    Result of citation validation.
    """

    passed: bool
    citations: List[CitationMeta]
    invalid_citations: List[str]


def build_context_map(chunks: Sequence[RerankedResult]) -> Dict[str, RerankedResult]:
    """
    Create a mapping from chunk_id -> chunk so we can validate citations quickly.
    """
    return {c.chunk_id: c for c in chunks}


def validate_citations(
    answer_text: str,
    context_chunks: Sequence[RerankedResult],
    *,
    preview_chars: int = 180,
) -> CitationValidationResult:
    """
    Validate citations found in `answer_text` against the provided context chunks.

    Rules:
    - Citations must be in the format [chunk_id]
    - Each cited chunk_id must exist in `context_chunks`
    - Duplicate citations are OK (we return each cited id once, in order)
    - If any cited id is missing, validation fails
    """
    cited_ids = extract_citations(answer_text)
    context_map = build_context_map(context_chunks)

    valid: List[CitationMeta] = []
    invalid: List[str] = []

    for cid in cited_ids:
        chunk = context_map.get(cid)
        if chunk is None:
            invalid.append(cid)
            continue

        preview = (chunk.text or "").replace("\n", " ").strip()
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "..."

        valid.append(
            CitationMeta(
                chunk_id=cid,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                text_preview=preview,
            )
        )

    passed = len(invalid) == 0
    return CitationValidationResult(passed=passed, citations=valid, invalid_citations=invalid)


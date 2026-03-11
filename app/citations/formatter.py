from __future__ import annotations

from typing import Iterable, List

from app.citations.validator import CitationMeta, CitationValidationResult


def format_citation(meta: CitationMeta) -> str:
    """
    Format a single validated citation for display in the terminal.
    """
    return (
        f"- [{meta.chunk_id}] {meta.source_file} (page {meta.page_number})\n"
        f"  preview: {meta.text_preview}"
    )


def format_validation_result(result: CitationValidationResult) -> str:
    """
    Format the overall validation result for display.
    """
    lines: List[str] = []
    lines.append(f"Validation passed: {result.passed}")

    if result.invalid_citations:
        lines.append("Invalid citations:")
        for cid in result.invalid_citations:
            lines.append(f"- [{cid}]")

    if result.citations:
        lines.append("Validated citations:")
        for meta in result.citations:
            lines.append(format_citation(meta))
    else:
        lines.append("Validated citations: (none)")

    return "\n".join(lines)


from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """
    A chunk of text produced by the ingestion pipeline.
    """

    chunk_id: str
    source_file: str
    page_number: int
    text: str


def split_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    This is intentionally beginner-friendly and dependency-free.

    Example:
    - chunk_size=10, overlap=2
    - chunk 1: text[0:10]
    - chunk 2 starts at 8 (10-2), etc.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    # Basic sliding window
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_page_text(
    *,
    source_file: str,
    page_number: int,
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Create `Chunk` objects for a single page of a single PDF.
    """
    parts = split_text(text, chunk_size=chunk_size, overlap=overlap)
    out: List[Chunk] = []
    for i, part in enumerate(parts):
        chunk_id = f"{source_file}-p{page_number}-c{i:04d}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page_number,
                text=part,
            )
        )
    return out


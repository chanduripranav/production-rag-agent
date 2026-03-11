from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_page_text, split_text


def test_split_text_empty() -> None:
    assert split_text("") == []


def test_split_text_invalid_args() -> None:
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=-1)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=10)


def test_split_text_overlap_behavior() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    parts = split_text(text, chunk_size=10, overlap=3)
    assert parts[0] == "abcdefghij"
    # Next chunk should start at 10-3 = 7 => "hij..." (3 chars overlap)
    assert parts[1].startswith("hij")


def test_chunk_page_text_fields() -> None:
    chunks = chunk_page_text(
        source_file="doc.pdf",
        page_number=2,
        text="0123456789" * 5,
        chunk_size=20,
        overlap=5,
    )
    assert len(chunks) >= 1
    c0 = chunks[0]
    assert c0.source_file == "doc.pdf"
    assert c0.page_number == 2
    assert c0.chunk_id.startswith("doc.pdf-p2-c")
    assert isinstance(c0.text, str) and len(c0.text) > 0


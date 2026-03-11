from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.bm25_index import BM25Index, ChunkRecord, load_chunks


def test_load_chunks_from_real_file() -> None:
    """
    Basic smoke test that the ingestion output can be loaded.
    """
    chunks_path = Path("data/processed/chunks.json")
    chunks = load_chunks(chunks_path)
    assert len(chunks) > 0
    assert chunks[0].chunk_id
    assert chunks[0].source_file
    assert chunks[0].page_number >= 1
    assert isinstance(chunks[0].text, str)


def test_bm25_search_returns_results_for_obvious_term() -> None:
    """
    Build a tiny in-memory corpus and ensure BM25 finds the relevant chunk.
    """
    chunks = [
        ChunkRecord(chunk_id="c1", source_file="a.pdf", page_number=1, text="Fourier transform is useful."),
        ChunkRecord(chunk_id="c2", source_file="b.pdf", page_number=2, text="Bundle adjustment and cameras."),
        ChunkRecord(chunk_id="c3", source_file="c.pdf", page_number=3, text="Snakes and active contours."),
    ]
    index = BM25Index(chunks)
    results = index.search("fourier transform", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score >= results[-1].score


def test_bm25_search_empty_query() -> None:
    chunks = [ChunkRecord(chunk_id="c1", source_file="a.pdf", page_number=1, text="Hello world")]
    index = BM25Index(chunks)
    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []


def test_bm25_search_top_k_zero() -> None:
    chunks = [ChunkRecord(chunk_id="c1", source_file="a.pdf", page_number=1, text="Hello world")]
    index = BM25Index(chunks)
    assert index.search("hello", top_k=0) == []


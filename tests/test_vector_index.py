from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from app.retrieval.bm25_index import ChunkRecord
from app.retrieval.vector_index import VectorIndex


@dataclass
class FakeEmbeddingModel:
    """
    A tiny deterministic embedding model for tests.

    It maps a string to a 2D vector:
    - dim0: count of 'a'
    - dim1: count of 'b'
    """

    def encode(self, sentences: Sequence[str], **kwargs):  # type: ignore[no-untyped-def]
        vectors: List[List[float]] = []
        for s in sentences:
            s = (s or "").lower()
            vectors.append([float(s.count("a")), float(s.count("b"))])
        return np.asarray(vectors, dtype=np.float32)


def test_vector_index_build_and_search_with_fake_model() -> None:
    chunks = [
        ChunkRecord(chunk_id="c1", source_file="a.pdf", page_number=1, text="aaaa"),
        ChunkRecord(chunk_id="c2", source_file="b.pdf", page_number=2, text="bbbb"),
        ChunkRecord(chunk_id="c3", source_file="ab.pdf", page_number=3, text="ab"),
    ]

    model = FakeEmbeddingModel()
    vindex = VectorIndex.build(chunks, embedding_model_name="fake-model", model=model)

    # Query with more 'a' should rank 'aaaa' highest.
    results = vindex.search("aaaa", top_k=3, model=model)
    assert len(results) == 3
    assert results[0].chunk_id == "c1"


def test_vector_index_top_k_zero() -> None:
    chunks = [ChunkRecord(chunk_id="c1", source_file="a.pdf", page_number=1, text="aaaa")]
    vindex = VectorIndex.build(chunks, embedding_model_name="fake-model", model=FakeEmbeddingModel())
    assert vindex.search("aaaa", top_k=0, model=FakeEmbeddingModel()) == []


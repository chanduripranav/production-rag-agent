from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.hybrid import HybridResult


@dataclass
class FakeCrossEncoder:
    """
    Fake cross-encoder used in tests.

    It scores a pair (query, text) by counting how many times the query token
    "fourier" appears in the text. (Deterministic and fast.)
    """

    def predict(self, sentences: Sequence[Tuple[str, str]], **kwargs):  # type: ignore[no-untyped-def]
        scores: List[float] = []
        for query, text in sentences:
            q = (query or "").lower()
            t = (text or "").lower()
            if "fourier" in q:
                scores.append(float(t.count("fourier")))
            else:
                scores.append(0.0)
        return np.asarray(scores, dtype=np.float32)


def test_reranker_sorts_by_score_desc() -> None:
    candidates = [
        HybridResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="no match", fused_score=0.1),
        HybridResult(chunk_id="c2", source_file="b.pdf", page_number=1, text="Fourier Fourier", fused_score=0.1),
        HybridResult(chunk_id="c3", source_file="c.pdf", page_number=1, text="Fourier", fused_score=0.1),
    ]

    reranker = CrossEncoderReranker(model_name="fake", model=FakeCrossEncoder())
    results = reranker.rerank("Fourier transform", candidates, top_k=3)

    assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]
    assert results[0].rerank_score >= results[1].rerank_score >= results[2].rerank_score


def test_reranker_top_k_zero() -> None:
    reranker = CrossEncoderReranker(model_name="fake", model=FakeCrossEncoder())
    candidates = [HybridResult(chunk_id="c1", source_file="a.pdf", page_number=1, text="Fourier", fused_score=0.1)]
    assert reranker.rerank("Fourier transform", candidates, top_k=0) == []


def test_reranker_empty_candidates() -> None:
    reranker = CrossEncoderReranker(model_name="fake", model=FakeCrossEncoder())
    assert reranker.rerank("Fourier transform", [], top_k=5) == []


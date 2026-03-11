from __future__ import annotations

from app.retrieval.bm25_index import SearchResult
from app.retrieval.hybrid import fuse_rrf, rrf_score
from app.retrieval.vector_index import VectorSearchResult


def test_rrf_score_is_higher_for_better_rank() -> None:
    assert rrf_score(1, k=60) > rrf_score(2, k=60)
    assert rrf_score(2, k=60) > rrf_score(10, k=60)


def test_fuse_rrf_overlapping_chunk_ids_sum_scores() -> None:
    bm25 = [
        SearchResult(chunk_id="A", source_file="a.pdf", page_number=1, text="aaa", score=10.0),
        SearchResult(chunk_id="B", source_file="b.pdf", page_number=2, text="bbb", score=9.0),
    ]
    vec = [
        VectorSearchResult(chunk_id="B", source_file="b.pdf", page_number=2, text="bbb", score=0.9),
        VectorSearchResult(chunk_id="C", source_file="c.pdf", page_number=3, text="ccc", score=0.8),
    ]

    fused = fuse_rrf(bm25, vec, k=60)
    fused_by_id = {r.chunk_id: r for r in fused}

    # Overlapping chunk "B" should have contributions from BOTH lists.
    expected_b = rrf_score(2, k=60) + rrf_score(1, k=60)
    assert abs(fused_by_id["B"].fused_score - expected_b) < 1e-9

    # Unique chunks should only have one contribution.
    assert abs(fused_by_id["A"].fused_score - rrf_score(1, k=60)) < 1e-9
    assert abs(fused_by_id["C"].fused_score - rrf_score(2, k=60)) < 1e-9


def test_fuse_rrf_ranking_prefers_agreement() -> None:
    # "X" is rank 1 in BM25, missing in vector.
    bm25 = [SearchResult(chunk_id="X", source_file="x.pdf", page_number=1, text="x", score=10.0)]

    # "Y" is rank 10 in BOTH BM25 and vector -> it gets two contributions.
    bm25 += [
        SearchResult(chunk_id=f"F{i}", source_file="f.pdf", page_number=1, text="f", score=0.0)
        for i in range(2, 10)
    ]
    bm25 += [SearchResult(chunk_id="Y", source_file="y.pdf", page_number=1, text="y", score=0.0)]

    vec = [
        VectorSearchResult(chunk_id=f"V{i}", source_file="v.pdf", page_number=1, text="v", score=0.0)
        for i in range(1, 10)
    ]
    vec += [VectorSearchResult(chunk_id="Y", source_file="y.pdf", page_number=1, text="y", score=0.0)]

    fused = fuse_rrf(bm25, vec, k=60)

    # "Y" should outrank "X" because it appears in both rankings.
    # (This illustrates the whole point of RRF.)
    assert fused[0].chunk_id == "Y"


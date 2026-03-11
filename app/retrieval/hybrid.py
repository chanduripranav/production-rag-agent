from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from app.retrieval.bm25_index import ChunkRecord, SearchResult, build_bm25_from_chunks_file
from app.retrieval.vector_index import VectorIndex, VectorSearchResult


@dataclass(frozen=True)
class HybridResult:
    """
    The final result returned by hybrid retrieval.

    `fused_score` is computed by Reciprocal Rank Fusion (RRF).
    """

    chunk_id: str
    source_file: str
    page_number: int
    text: str
    fused_score: float


def rrf_score(rank: int, *, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion (RRF) contribution for a single ranking.

    - `rank` is 1-based (rank 1 is the top result).
    - `k` is a small constant (often 60) that reduces the impact of very high ranks.
    """
    return 1.0 / (k + rank)


def fuse_rrf(
    bm25_results: Sequence[SearchResult],
    vector_results: Sequence[VectorSearchResult],
    *,
    k: int = 60,
) -> List[HybridResult]:
    """
    Fuse BM25 + vector rankings using RRF.

    Important:
    - We use ranks, not raw scores.
    - If a chunk appears in both lists, its fused score increases.
    """
    # Map chunk_id -> (ChunkRecord-like data, fused_score)
    fused: Dict[str, Tuple[ChunkRecord, float]] = {}

    def _add(chunk: ChunkRecord, add_score: float) -> None:
        if chunk.chunk_id in fused:
            prev_chunk, prev_score = fused[chunk.chunk_id]
            fused[chunk.chunk_id] = (prev_chunk, prev_score + add_score)
        else:
            fused[chunk.chunk_id] = (chunk, add_score)

    # BM25 contribution
    for rank, r in enumerate(bm25_results, start=1):
        chunk = ChunkRecord(
            chunk_id=r.chunk_id,
            source_file=r.source_file,
            page_number=r.page_number,
            text=r.text,
        )
        _add(chunk, rrf_score(rank, k=k))

    # Vector contribution
    for rank, r in enumerate(vector_results, start=1):
        chunk = ChunkRecord(
            chunk_id=r.chunk_id,
            source_file=r.source_file,
            page_number=r.page_number,
            text=r.text,
        )
        _add(chunk, rrf_score(rank, k=k))

    # Sort by fused score (descending)
    sorted_items = sorted(fused.values(), key=lambda item: item[1], reverse=True)
    return [
        HybridResult(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            text=chunk.text,
            fused_score=score,
        )
        for chunk, score in sorted_items
    ]


class HybridRetriever:
    """
    A small orchestrator that runs BM25 + vector search, then fuses results.
    """

    def __init__(self, *, chunks_path: Path, vector_index_path: Path, vector_meta_path: Path):
        self._chunks_path = chunks_path
        self._vector_index_path = vector_index_path
        self._vector_meta_path = vector_meta_path

        # BM25 is small enough to build quickly in-memory for now.
        self._bm25 = build_bm25_from_chunks_file(chunks_path)

        # Vector index must be built beforehand (see scripts/build_vector_index.py).
        self._vector = VectorIndex.load(index_path=vector_index_path, meta_path=vector_meta_path)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        bm25_top_k: int = 25,
        vector_top_k: int = 25,
        rrf_k: int = 60,
    ) -> List[HybridResult]:
        """
        Run both retrievers and fuse their results using RRF.

        `bm25_top_k` and `vector_top_k` can be larger than `top_k` so fusion has more candidates.
        """
        bm25_results = self._bm25.search(query, top_k=bm25_top_k)
        vector_results = self._vector.search(query, top_k=vector_top_k)

        fused = fuse_rrf(bm25_results, vector_results, k=rrf_k)
        return fused[:top_k]


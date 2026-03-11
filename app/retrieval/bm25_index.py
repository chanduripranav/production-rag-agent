from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi

from app.retrieval.tokenizer import tokenize


@dataclass(frozen=True)
class ChunkRecord:
    """
    A single chunk as stored in data/processed/chunks.json.
    """

    chunk_id: str
    source_file: str
    page_number: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """
    A chunk + its BM25 score.
    """

    chunk_id: str
    source_file: str
    page_number: int
    text: str
    score: float


def load_chunks(chunks_path: Path) -> List[ChunkRecord]:
    """
    Load chunk data from the JSON produced by the ingestion pipeline.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks: List[ChunkRecord] = []

    for item in raw:
        chunks.append(
            ChunkRecord(
                chunk_id=str(item["chunk_id"]),
                source_file=str(item["source_file"]),
                page_number=int(item["page_number"]),
                text=str(item["text"]),
            )
        )

    return chunks


class BM25Index:
    """
    A small wrapper around rank-bm25 to keep the rest of the codebase clean.
    """

    def __init__(self, chunks: Sequence[ChunkRecord]):
        self._chunks: List[ChunkRecord] = list(chunks)

        # Tokenize each chunk once and store it.
        self._tokenized_corpus: List[List[str]] = [tokenize(c.text) for c in self._chunks]

        # Build BM25 index.
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, top_k: int = 5) -> List[SearchResult]:
        """
        Search the chunk corpus with BM25 and return the top results.
        """
        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # rank-bm25 returns a score per document in the same order as the corpus.
        scores = self._bm25.get_scores(query_tokens)

        # Get top_k indices by score (descending).
        ranked_idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]

        results: List[SearchResult] = []
        for i in ranked_idxs:
            chunk = self._chunks[i]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(scores[i]),
                )
            )

        return results


def build_bm25_from_chunks_file(chunks_path: Path) -> BM25Index:
    """
    Convenience function used by scripts/tests.
    """
    chunks = load_chunks(chunks_path)
    return BM25Index(chunks)


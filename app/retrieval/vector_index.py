from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from app.retrieval.bm25_index import ChunkRecord, load_chunks
from app.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    EmbeddingModel,
    embed_texts,
    load_embedding_model,
    normalize_vectors,
)


@dataclass(frozen=True)
class VectorIndexMeta:
    """
    Metadata we save next to the FAISS index so we can recover chunk info.
    """

    embedding_model: str
    chunks: List[ChunkRecord]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    score: float


class VectorIndex:
    """
    A simple vector index built with FAISS.

    We use cosine similarity by:
    - normalizing embeddings to unit length
    - using an inner-product index (IndexFlatIP)
    """

    def __init__(self, *, index: faiss.Index, meta: VectorIndexMeta):
        self._index = index
        self._meta = meta

    @property
    def size(self) -> int:
        return len(self._meta.chunks)

    @property
    def embedding_model(self) -> str:
        return self._meta.embedding_model

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        model: EmbeddingModel | None = None,
    ) -> List[VectorSearchResult]:
        """
        Vector search over the chunks.
        """
        if top_k <= 0:
            return []

        # Allow passing a model (useful for tests). Otherwise load by name.
        embedder = model or load_embedding_model(EmbeddingConfig(model_name=self._meta.embedding_model))
        q = embed_texts(embedder, [query])
        q = normalize_vectors(q)

        # FAISS expects float32 numpy arrays.
        scores, idxs = self._index.search(q.astype(np.float32), top_k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[VectorSearchResult] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            chunk = self._meta.chunks[int(i)]
            results.append(
                VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(score),
                )
            )

        return results

    @staticmethod
    def build(
        chunks: Sequence[ChunkRecord],
        *,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        model: EmbeddingModel | None = None,
    ) -> "VectorIndex":
        """
        Build a VectorIndex from chunks.

        - `model` can be passed in (useful for tests).
        - Otherwise we load SentenceTransformer by name.
        """
        chunk_list = list(chunks)
        texts = [c.text for c in chunk_list]

        embedder = model or load_embedding_model(EmbeddingConfig(model_name=embedding_model_name))
        vectors = embed_texts(embedder, texts)
        vectors = normalize_vectors(vectors)

        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_list):
            raise ValueError("Unexpected embedding shape returned by the embedding model.")

        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))

        meta = VectorIndexMeta(embedding_model=embedding_model_name, chunks=chunk_list)
        return VectorIndex(index=index, meta=meta)

    def save(self, *, index_path: Path, meta_path: Path) -> None:
        """
        Save the FAISS index + metadata to disk.
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))

        meta_dict = {
            "embedding_model": self._meta.embedding_model,
            "chunks": [asdict(c) for c in self._meta.chunks],
        }
        meta_path.write_text(json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(*, index_path: Path, meta_path: Path) -> "VectorIndex":
        """
        Load the FAISS index + metadata from disk.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata file not found: {meta_path}")

        index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))

        chunks: List[ChunkRecord] = []
        for item in raw["chunks"]:
            chunks.append(
                ChunkRecord(
                    chunk_id=str(item["chunk_id"]),
                    source_file=str(item["source_file"]),
                    page_number=int(item["page_number"]),
                    text=str(item["text"]),
                )
            )

        meta = VectorIndexMeta(embedding_model=str(raw["embedding_model"]), chunks=chunks)
        return VectorIndex(index=index, meta=meta)


def build_and_save_vector_index(
    *,
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VectorIndex:
    """
    Convenience function for scripts.
    """
    chunks = load_chunks(chunks_path)
    vindex = VectorIndex.build(chunks, embedding_model_name=embedding_model_name)
    vindex.save(index_path=index_path, meta_path=meta_path)
    return vindex


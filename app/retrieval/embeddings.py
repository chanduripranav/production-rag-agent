from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """
    A tiny interface so we can plug in a fake model in tests.
    """

    def encode(self, sentences: Sequence[str], **kwargs):  # type: ignore[no-untyped-def]
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for embedding.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL


def load_embedding_model(config: EmbeddingConfig | None = None) -> SentenceTransformer:
    """
    Load the default SentenceTransformer model.

    The first time you run this, it will download the model.
    """
    cfg = config or EmbeddingConfig()
    return SentenceTransformer(cfg.model_name)


def embed_texts(model: EmbeddingModel, texts: List[str]) -> np.ndarray:
    """
    Embed texts into a 2D numpy array (shape: [n_texts, dim]).

    We explicitly request numpy output and float32 for FAISS.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll normalize ourselves for clarity
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.

    This lets us use inner product in FAISS as cosine similarity.
    """
    if vectors.size == 0:
        return vectors.astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for empty/degenerate vectors.
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


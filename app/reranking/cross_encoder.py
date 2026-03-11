from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from app.retrieval.hybrid import HybridResult


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderModel(Protocol):
    """
    Small interface so tests can pass a fake model.
    """

    def predict(self, sentences: Sequence[Tuple[str, str]], **kwargs):  # type: ignore[no-untyped-def]
        ...


@dataclass(frozen=True)
class RerankedResult:
    """
    Output of the reranker.
    """

    chunk_id: str
    source_file: str
    page_number: int
    text: str
    rerank_score: float


def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> CrossEncoder:
    """
    Load the default cross-encoder model.

    The first run will download the model.
    """
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """
    Rerank candidate chunks using a cross-encoder model.

    Conceptually:
    - hybrid retrieval gives us a candidate list
    - cross-encoder scores (query, chunk_text) pairs
    - we sort by cross-encoder score
    """

    def __init__(self, *, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, model: CrossEncoderModel | None = None):
        self._model_name = model_name
        self._model = model  # if None we lazily load a real model

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            self._model = load_cross_encoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridResult],
        *,
        top_k: int = 5,
    ) -> List[RerankedResult]:
        """
        Rerank the provided candidates and return the top results.
        """
        if top_k <= 0:
            return []
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._get_model().predict(pairs)
        scores = np.asarray(scores, dtype=np.float32)

        ranked_idxs = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)[:top_k]

        out: List[RerankedResult] = []
        for i in ranked_idxs:
            c = candidates[i]
            out.append(
                RerankedResult(
                    chunk_id=c.chunk_id,
                    source_file=c.source_file,
                    page_number=c.page_number,
                    text=c.text,
                    rerank_score=float(scores[i]),
                )
            )
        return out


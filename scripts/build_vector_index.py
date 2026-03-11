from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/build_vector_index.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.retrieval.vector_index import build_and_save_vector_index  # noqa: E402


def main() -> None:
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    index_path = ROOT / "data" / "processed" / "vector.index"
    meta_path = ROOT / "data" / "processed" / "vector_meta.json"

    vindex = build_and_save_vector_index(
        chunks_path=chunks_path,
        index_path=index_path,
        meta_path=meta_path,
    )

    print(f"Chunks: {vindex.size}")
    print(f"Embedding model: {vindex.embedding_model}")
    print(f"Saved FAISS index: {index_path}")
    print(f"Saved metadata:   {meta_path}")


if __name__ == "__main__":
    main()


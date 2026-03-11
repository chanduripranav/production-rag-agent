from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/search_vector.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.retrieval.vector_index import VectorIndex  # noqa: E402


def main() -> None:
    # Hardcoded query for now (per requirements).
    query = "What is Fourier transform?"

    index_path = ROOT / "data" / "processed" / "vector.index"
    meta_path = ROOT / "data" / "processed" / "vector_meta.json"

    vindex = VectorIndex.load(index_path=index_path, meta_path=meta_path)
    results = vindex.search(query, top_k=5)

    print("=" * 80)
    print("Vector Search Demo (FAISS + SentenceTransformers)")
    print(f"Query: {query}")
    print(f"Chunks indexed: {vindex.size}")
    print(f"Embedding model: {vindex.embedding_model}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\n[{rank}] score={r.score:.4f}")
        print(f"chunk_id   : {r.chunk_id}")
        print(f"source_file: {r.source_file}")
        print(f"page       : {r.page_number}")
        print("text:")
        preview = r.text.replace("\n", " ").strip()
        if len(preview) > 400:
            preview = preview[:400] + "..."
        print(preview)


if __name__ == "__main__":
    main()


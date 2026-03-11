from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/search_hybrid.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.retrieval.hybrid import HybridRetriever  # noqa: E402


def main() -> None:
    # Hardcoded query for now (per requirements).
    query = "What is Fourier transform?"

    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    vector_index_path = ROOT / "data" / "processed" / "vector.index"
    vector_meta_path = ROOT / "data" / "processed" / "vector_meta.json"

    retriever = HybridRetriever(
        chunks_path=chunks_path,
        vector_index_path=vector_index_path,
        vector_meta_path=vector_meta_path,
    )
    results = retriever.search(query, top_k=5)

    print("=" * 80)
    print("Hybrid Search Demo (BM25 + Vector, fused with RRF)")
    print(f"Query: {query}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\n[{rank}] fused_score={r.fused_score:.6f}")
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


from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/search_bm25.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.retrieval.bm25_index import build_bm25_from_chunks_file  # noqa: E402


def main() -> None:
    # Hardcoded query for now (per requirements).
    query = "What is Fourier transform?"

    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    index = build_bm25_from_chunks_file(chunks_path)
    results = index.search(query, top_k=5)

    print("=" * 80)
    print("BM25 Search Demo")
    print(f"Query: {query}")
    print(f"Chunks indexed: {index.size}")
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
        # Print a short preview so the terminal stays readable.
        preview = r.text.replace("\n", " ").strip()
        if len(preview) > 400:
            preview = preview[:400] + "..."
        print(preview)


if __name__ == "__main__":
    main()


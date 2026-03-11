from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/ingest.py` to import from `app/` without extra setup.
# (In a packaged app you might rely on `pip install -e .` instead.)
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.ingestion.chunker import chunk_page_text  # noqa: E402
from app.ingestion.loader import list_pdfs  # noqa: E402
from app.ingestion.parser import parse_pdf_pages  # noqa: E402


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    output_path = processed_dir / "chunks.json"

    processed_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list_pdfs(raw_dir)
    all_chunks = []

    for doc in pdfs:
        for page in parse_pdf_pages(doc.path):
            # Chunk each page separately so we can keep `page_number` accurate.
            chunks = chunk_page_text(
                source_file=doc.source_file,
                page_number=page.page_number,
                text=page.text,
                chunk_size=1200,
                overlap=200,
            )
            all_chunks.extend(chunks)

    # Save as a JSON list of dicts.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(f"Found PDFs: {len(pdfs)}")
    print(f"Wrote chunks: {len(all_chunks)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()


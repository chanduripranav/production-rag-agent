from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DocumentFile:
    """
    A tiny container describing a source document on disk.

    Keeping this explicit makes it easier to attach extra metadata later
    (like a doc_id, collection name, etc.) without changing the whole pipeline.
    """

    path: Path

    @property
    def source_file(self) -> str:
        # Store just the filename in output JSON to keep it portable.
        return self.path.name


def list_pdfs(raw_dir: Path) -> List[DocumentFile]:
    """
    Find all PDF files under data/raw (recursively).

    - Uses pathlib throughout
    - Returns a stable sorted list for reproducibility
    """
    if not raw_dir.exists():
        return []

    pdf_paths = sorted(p for p in raw_dir.rglob("*.pdf") if p.is_file())
    return [DocumentFile(path=p) for p in pdf_paths]


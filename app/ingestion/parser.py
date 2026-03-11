from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ParsedPage:
    """
    A single PDF page worth of extracted text.
    """

    page_number: int  # 1-based
    text: str


def parse_pdf_pages(pdf_path: Path) -> Iterator[ParsedPage]:
    """
    Extract text from a PDF page-by-page using PyMuPDF.

    Notes:
    - Page numbers are returned as 1-based (humans expect this).
    - We keep text extraction simple: `page.get_text()` with default settings.
    """
    with fitz.open(pdf_path) as doc:
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            text = page.get_text() or ""
            yield ParsedPage(page_number=idx + 1, text=text.strip())


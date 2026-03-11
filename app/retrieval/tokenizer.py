from __future__ import annotations

import re
from typing import List


# A simple "word" regex:
# - keeps letters and digits
# - splits on punctuation/whitespace
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Turn free-form text into a list of tokens.

    Beginner-friendly approach:
    - lowercase everything
    - keep only letters and digits
    - ignore punctuation
    """
    if not text:
        return []

    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


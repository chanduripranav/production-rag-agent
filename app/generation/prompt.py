from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from app.reranking.cross_encoder import RerankedResult


@dataclass(frozen=True)
class PromptParts:
    """
    Keeping the prompt parts separate makes it easier to test and modify.
    """

    system_instructions: str
    context: str
    question: str

    def render(self) -> str:
        return f"{self.system_instructions}\n\n{self.context}\n\n{self.question}\n"


def build_grounded_prompt(question: str, chunks: Sequence[RerankedResult]) -> PromptParts:
    """
    Build a strict grounded prompt.

    Requirements baked into the instructions:
    - Answer ONLY using the provided context
    - Use citations in the format [chunk_id]
    - If insufficient context, return the refusal sentence verbatim
    """
    system_instructions = (
        "You are a careful assistant for an 'Ask My Docs' system.\n"
        "You MUST follow these rules:\n"
        "1) Answer ONLY using the provided CONTEXT. Do not use outside knowledge.\n"
        "2) Every factual claim must include citations in the form [chunk_id].\n"
        "3) Do NOT invent citations.\n"
        "4) If the CONTEXT does not contain enough information to answer, reply EXACTLY with:\n"
        '   "I could not find enough support in the indexed documents."\n'
        "5) Keep the answer clear and concise.\n"
    )

    if not chunks:
        context = "CONTEXT:\n(no chunks provided)\n"
    else:
        lines: List[str] = ["CONTEXT:"]
        for c in chunks:
            # Include the chunk_id prominently so it is easy to cite.
            lines.append(f"\n[chunk_id={c.chunk_id}] source={c.source_file} page={c.page_number}")
            lines.append(c.text.strip())
        context = "\n".join(lines) + "\n"

    question_block = f"QUESTION:\n{question.strip()}\n\nANSWER:"
    return PromptParts(system_instructions=system_instructions, context=context, question=question_block)


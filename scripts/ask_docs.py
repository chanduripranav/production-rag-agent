from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/ask_docs.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.generation.generator import GroundedAnswerGenerator, MockLLMClient  # noqa: E402
from app.generation.prompt import build_grounded_prompt  # noqa: E402
from app.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402


def main() -> None:
    # Hardcoded query for now (per requirements).
    question = "What is Fourier transform?"

    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    vector_index_path = ROOT / "data" / "processed" / "vector.index"
    vector_meta_path = ROOT / "data" / "processed" / "vector_meta.json"

    # 1) Hybrid retrieval
    hybrid = HybridRetriever(
        chunks_path=chunks_path,
        vector_index_path=vector_index_path,
        vector_meta_path=vector_meta_path,
    )
    candidates = hybrid.search(question, top_k=25)

    # 2) Rerank
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(question, candidates, top_k=5)

    # 3) Build prompt (useful to inspect/debug)
    prompt_parts = build_grounded_prompt(question, reranked)
    prompt = prompt_parts.render()

    # 4) Generate answer (using mock LLM by default)
    generator = GroundedAnswerGenerator(llm=MockLLMClient())
    result = generator.generate(question, reranked)

    print("=" * 80)
    print("Ask My Docs (Grounded Answer Generation Demo)")
    print("=" * 80)
    print(f"\nQuestion:\n{question}")

    print("\nFinal answer:\n" + result.answer)

    print("\nCitations used:")
    if result.citations:
        for c in result.citations:
            print(f"- [{c}]")
    else:
        print("(none)")

    print("\nSource chunk previews:")
    for c in reranked:
        preview = (c.text or "").replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"\n- chunk_id={c.chunk_id} source={c.source_file} page={c.page_number}")
        print(f"  preview: {preview}")

    # Optional: show the prompt if you want to debug what gets sent to an LLM.
    # print("\n--- PROMPT (debug) ---\n")
    # print(prompt)


if __name__ == "__main__":
    main()


from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/ask_docs_validated.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.citations.formatter import format_validation_result  # noqa: E402
from app.citations.validator import validate_citations  # noqa: E402
from app.generation.generator import GroundedAnswerGenerator, MockLLMClient  # noqa: E402
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

    # 3) Generate answer (mock by default)
    generator = GroundedAnswerGenerator(llm=MockLLMClient())
    gen = generator.generate(question, reranked)

    # 4) Validate citations against the reranked context
    validation = validate_citations(gen.answer, reranked)

    print("=" * 80)
    print("Ask My Docs (Validated Citations Demo)")
    print("=" * 80)
    print(f"\nQuestion:\n{question}")
    print("\nAnswer:\n" + gen.answer)
    print("\n" + format_validation_result(validation))


if __name__ == "__main__":
    main()


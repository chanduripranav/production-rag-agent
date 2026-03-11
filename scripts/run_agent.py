from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """
    Return the repo root (the parent of the `scripts/` folder).
    """
    return Path(__file__).resolve().parents[1]


# Allow `python scripts/run_agent.py` to import from `app/`.
ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.retrieval.hybrid import HybridRetriever
from app.reranking.cross_encoder import CrossEncoderReranker
from app.generation.generator import GroundedAnswerGenerator, MockLLMClient
from app.agent.workflow import execute_agent


def main() -> None:
    # 1. Hardcoded sample query
    query = "What is Fourier transform?"
    
    # 2. Setup paths
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    vector_index_path = ROOT / "data" / "processed" / "vector.index"
    vector_meta_path = ROOT / "data" / "processed" / "vector_meta.json"
    
    # 3. Initialize components
    print("Initializing Agent Components...")
    retriever = HybridRetriever(
        chunks_path=chunks_path,
        vector_index_path=vector_index_path,
        vector_meta_path=vector_meta_path,
    )
    reranker = CrossEncoderReranker()
    llm = MockLLMClient()
    generator = GroundedAnswerGenerator(llm=llm)
    
    print("-" * 80)
    print("Running Agent Worklow...")
    print("-" * 80)
    
    # 4. Execute the agent workflow
    result = execute_agent(query, retriever, reranker, generator)
    
    # 5. Print out the trace
    print(f"Query                      : {result['query']}")
    print(f"Detected Query Type        : {result['query_type']}")
    print(f"Retrieval Strategy Used    : bm25_top_k={result['strategy'].get('bm25_top_k')}, vector_top_k={result['strategy'].get('vector_top_k')}")
    print(f"Evidence Sufficient?       : {'Yes' if result['is_sufficient'] else 'No'}")
    print("-" * 80)
    print("Final Answer:")
    print(result['answer'])
    print("-" * 80)
    
    if result.get("citation_validation"):
        cv = result["citation_validation"]
        print(f"Citation Validation Result : {'PASSED' if cv.passed else 'FAILED'}")
        if cv.invalid_citations:
            print(f"  Invalid citations detected: {cv.invalid_citations}")
    else:
        print("Citation Validation Result : None")
        
    print("Citations:")
    if result.get("citations"):
        for c in result["citations"]:
            print(f"  - [{c}]")
    else:
        print("  None")
        
    print("=" * 80)


if __name__ == "__main__":
    main()

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.hybrid import HybridRetriever
from app.evaluation.retrieval_eval import load_eval_set, evaluate_retrieval

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--eval_file", type=str, default="data/eval/eval_set.json", help="Path to evaluation dataset")
    parser.add_argument("--chunks_file", type=str, default="data/processed/chunks.json", help="Path to chunk definitions")
    parser.add_argument("--vector_index", type=str, default="data/processed/vector.index", help="Path to vector index")
    parser.add_argument("--vector_meta", type=str, default="data/processed/vector_meta.json", help="Path to vector metadata")
    parser.add_argument("--recall_threshold", type=float, default=0.50, help="Minimum acceptable Recall@k threshold")
    parser.add_argument("--mrr_threshold", type=float, default=0.40, help="Minimum acceptable MRR threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_file = Path(args.eval_file)
    chunks_file = Path(args.chunks_file)
    vector_index = Path(args.vector_index)
    vector_meta = Path(args.vector_meta)

    # Check that required files exist
    if not eval_file.exists():
        print(f"ERROR: Dataset not found {eval_file}")
        sys.exit(1)
        
    for p in [chunks_file, vector_index, vector_meta]:
        if not p.exists():
            print(f"ERROR: Required data file missing {p}")
            print("Please ensure you've built the indices first.")
            sys.exit(1)

    print("Loading Hybrid Retriever...")
    retriever = HybridRetriever(
        chunks_path=chunks_file,
        vector_index_path=vector_index,
        vector_meta_path=vector_meta
    )

    print(f"Loading Evaluation Set from {eval_file}...")
    eval_cases = load_eval_set(eval_file)

    print(f"Evaluating {len(eval_cases)} cases...")
    metrics = evaluate_retrieval(retriever=retriever, eval_cases=eval_cases, k=5)

    recall = metrics.get('recall_at_k', 0.0)
    mrr = metrics.get('mrr', 0.0)
    ndcg = metrics.get('ndcg_at_k', 0.0)

    print("\n================== Evaluation Report ==================")
    print(f"Recall@5:  {recall:.4f}")
    print(f"MRR:       {mrr:.4f}")
    print(f"nDCG@5:    {ndcg:.4f}")
    print("=======================================================")

    success = True
    if recall < args.recall_threshold:
        print(f"❌ Failed: Recall@5 ({recall:.4f}) is below threshold ({args.recall_threshold})")
        success = False
    
    if mrr < args.mrr_threshold:
        print(f"❌ Failed: MRR ({mrr:.4f}) is below threshold ({args.mrr_threshold})")
        success = False
        
    if success:
        print("✅ Success: All thresholds met!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

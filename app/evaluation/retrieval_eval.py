import json
from pathlib import Path
from typing import Dict, List, Any

from app.retrieval.hybrid import HybridRetriever
from app.evaluation.metrics import recall_at_k, mrr, ndcg_at_k

def load_eval_set(filepath: Path) -> List[Dict[str, Any]]:
    """Load the evaluation cases from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_retrieval(
    retriever: HybridRetriever,
    eval_cases: List[Dict[str, Any]],
    k: int = 5
) -> Dict[str, float]:
    """
    Run the hybrid retriever over a list of evaluation cases and return average metrics.
    """
    if not eval_cases:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    for case in eval_cases:
        question = case["question"]
        expected_ids = case["expected_chunk_ids"]

        # Run retriever (get top K results)
        hyb_results = retriever.search(question, top_k=k)
        retrieved_ids = [res.chunk_id for res in hyb_results]

        # Calculate metrics for this case
        total_recall += recall_at_k(retrieved_ids, expected_ids, k=k)
        total_mrr += mrr(retrieved_ids, expected_ids)
        total_ndcg += ndcg_at_k(retrieved_ids, expected_ids, k=k)

    num_cases = len(eval_cases)
    
    return {
        "recall_at_k": total_recall / num_cases,
        "mrr": total_mrr / num_cases,
        "ndcg_at_k": total_ndcg / num_cases
    }

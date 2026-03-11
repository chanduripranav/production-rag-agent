from typing import List
import math

def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Calculate Recall@K: The fraction of expected chunk IDs that are successfully
    retrieved within the top K results.
    """
    if not expected_ids:
        return 0.0
    
    # Consider only top K retrieved items
    top_k_retrieved = retrieved_ids[:k]
    
    # Count how many expected items are in the top K retrieved
    hits = sum(1 for exp_id in expected_ids if exp_id in top_k_retrieved)
    
    return hits / len(expected_ids)


def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): 1 / rank of the first relevant document.
    """
    if not expected_ids or not retrieved_ids:
        return 0.0

    # Find the rank (1-indexed) of the first retrieved id that is expected
    for rank, ret_id in enumerate(retrieved_ids, start=1):
        if ret_id in expected_ids:
            return 1.0 / rank
            
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K).
    Here, relevance is binary (1 if retrieved doc is expected, 0 otherwise).
    """
    if not expected_ids:
        return 0.0

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, ret_id in enumerate(retrieved_ids[:k], start=1):
        if ret_id in expected_ids:
            # log base 2 of (rank + 1)
            dcg += 1.0 / math.log2(i + 1)
            
    # Calculate Ideal DCG (IDCG), which is the DCG if all expected items were at the top
    idcg = 0.0
    ideal_hits = min(len(expected_ids), k)
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

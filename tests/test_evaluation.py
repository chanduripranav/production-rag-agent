import math
import pytest
from app.evaluation.metrics import recall_at_k, mrr, ndcg_at_k

def test_recall_at_k():
    expected_ids = ["doc1", "doc2"]
    
    # Perfect match at top 5
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    assert recall_at_k(retrieved, expected_ids, k=5) == 1.0
    
    # 50% match at top 5
    retrieved = ["doc1", "doc3", "doc4", "doc5", "doc6"]
    assert recall_at_k(retrieved, expected_ids, k=5) == 0.5
    
    # 0% match at top 5
    retrieved = ["doc3", "doc4", "doc5", "doc6", "doc7"]
    assert recall_at_k(retrieved, expected_ids, k=5) == 0.0
    
    # Perfect match but lower rank
    retrieved = ["doc3", "doc4", "doc1", "doc2"]
    # For k=2, only doc3 and doc4 are considered
    assert recall_at_k(retrieved, expected_ids, k=2) == 0.0
    # For k=4, both doc1, doc2 are found
    assert recall_at_k(retrieved, expected_ids, k=4) == 1.0


def test_mrr():
    expected_ids = ["doc_relevant"]
    
    # Rank 1 hit
    assert mrr(["doc_relevant", "doc1", "doc2"], expected_ids) == 1.0
    # Rank 2 hit
    assert mrr(["doc1", "doc_relevant", "doc2"], expected_ids) == 0.5
    # Rank 3 hit
    assert mrr(["doc1", "doc2", "doc_relevant"], expected_ids) == pytest.approx(0.333, 0.01)
    # No hit
    assert mrr(["doc1", "doc2", "doc3"], expected_ids) == 0.0


def test_ndcg_at_k():
    expected_ids = ["doc1"]
    
    # Hit at rank 1 gives maximum possible gain, and matches ideal gain
    ret1 = ["doc1", "doc2", "doc3"]
    assert ndcg_at_k(ret1, expected_ids, k=3) == 1.0
    
    # Hit at rank 2 gives lower score
    ret2 = ["doc2", "doc1", "doc3"]
    score2 = ndcg_at_k(ret2, expected_ids, k=3)
    assert score2 > 0.0 and score2 < 1.0
    
    # Rank 2 DCG should be exactly: (1.0 / log2(2+1)) / (1.0 / log2(1+1))
    expected_score2 = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(score2, expected_score2)

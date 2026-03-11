from unittest.mock import MagicMock
from app.agent.planner import classify_query, choose_strategy
from app.agent.workflow import execute_agent, REFUSAL_MESSAGE
from app.reranking.cross_encoder import RerankedResult
from app.generation.generator import GenerationResult
from app.citations.validator import CitationValidationResult

def test_classify_query():
    assert classify_query('What is a neural network?') == 'conceptual'
    assert classify_query('Explain attention mechanism.') == 'conceptual'
    assert classify_query('hi there') == 'vague'
    assert classify_query('"machine learning" basics') == 'keyword'
    assert classify_query('IBM meaning') == 'keyword'

def test_choose_strategy():
    strategy_kw = choose_strategy('keyword')
    assert strategy_kw['bm25_top_k'] == 50
    assert strategy_kw['vector_top_k'] == 10

    strategy_vague = choose_strategy('vague')
    assert strategy_vague['bm25_top_k'] == 10

    strategy_concept = choose_strategy('conceptual')
    assert strategy_concept['vector_top_k'] == 40
    assert strategy_concept['bm25_top_k'] == 20

def test_execute_agent_sufficient_evidence():
    retriever_mock = MagicMock()
    reranker_mock = MagicMock()
    generator_mock = MagicMock()

    # Mock retrieval to return dummy candidates
    retriever_mock.search.return_value = ["dummy_candidate"]

    # Mock reranking to return a high score result
    high_score_result = RerankedResult(
        chunk_id="chunk1",
        source_file="file.pdf",
        page_number=1,
        text="dummy text",
        rerank_score=0.9
    )
    reranker_mock.rerank.return_value = [high_score_result]

    # Mock generation
    gen_result = GenerationResult(answer="A grounded answer [chunk1]", citations=["chunk1"])
    generator_mock.generate.return_value = gen_result

    # Execute
    query = "Explain test"
    result = execute_agent(query, retriever_mock, reranker_mock, generator_mock)

    assert result["is_sufficient"] is True
    assert result["answer"] == "A grounded answer [chunk1]"
    assert "chunk1" in result["citations"]
    assert retriever_mock.search.call_count == 1
    assert reranker_mock.rerank.call_count == 1
    assert generator_mock.generate.call_count == 1

def test_execute_agent_insufficient_evidence():
    retriever_mock = MagicMock()
    reranker_mock = MagicMock()
    generator_mock = MagicMock()

    # Mock retrieval to return dummy candidates
    retriever_mock.search.return_value = ["dummy_candidate"]

    # Mock reranking to return a LOW score result (first and fallback both weak)
    low_score_result = RerankedResult(
        chunk_id="chunk1",
        source_file="file.pdf",
        page_number=1,
        text="dummy text",
        rerank_score=-1.0 # Very low score, below 0.0 threshold
    )
    reranker_mock.rerank.return_value = [low_score_result]

    # Execute
    query = "Something random"
    result = execute_agent(query, retriever_mock, reranker_mock, generator_mock)

    assert result["is_sufficient"] is False
    assert result["answer"] == REFUSAL_MESSAGE
    # It should have called search twice (initial + fallback)
    assert retriever_mock.search.call_count == 2
    assert reranker_mock.rerank.call_count == 2
    # It should NOT call generator if evidence is insufficient
    assert generator_mock.generate.call_count == 0

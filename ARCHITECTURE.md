# 🏛️ System Architecture

This document provides a deep dive into the internal design of the **Agentic RAG System**. To maintain modularity, testability, and clarity, the codebase is separated into distinct, decoupled subsystems. 

---

## 1. Ingestion (Document Processing)
*Location: `app/ingestion/`*

The ingestion pipeline is responsible for transforming raw PDFs and Word documents into optimized, searchable indices. 
- **Parsing**: Extracts raw text while maintaining crucial metadata (like source filenames and page numbers).
- **Chunking**: Splits the extracted text into overlapping segments (chunks). Optimal chunk sizes preserve semantic meaning while ensuring context fits within the embedding and LLM token limits.
- **Indexing**: 
  - **BM25**: Creates an inverted index for exact-keyword lexical matching.
  - **Vector**: Uses Sentence-Transformers to generate dense embeddings for each chunk, which are stored in a FAISS index for high-speed similarity search.

## 2. Retrieval Strategies
*Location: `app/retrieval/`*

Relying on a single retrieval method often fails on complex queries. This system implements multiple strategies:
- **Lexical (BM25)**: Excellent for specific nouns, exact phrasing, or domain-specific terminology.
- **Semantic (Vector)**: Excels at understanding the *intent* of a query, even if synonyms are used.
- **Hybrid Fusion**: Executes both BM25 and Vector searches concurrently. It then merges the candidate lists using **Reciprocal Rank Fusion (RRF)**, ensuring chunks that score high in both methods are pushed to the very top.

## 3. Reranking
*Location: `app/reranking/`*

Even the best hybrid retrieval can return somewhat noisy results. The reranking layer acts as a highly accurate filter.
- **Cross-Encoder**: Instead of calculating similarity mathematically, a Cross-Encoder passes both the User Query and the Retrieved Chunk into a transformer model simultaneously. It outputs a highly accurate relevance score. The top `N` highest-scoring chunks are then passed to the generation phase.

## 4. Generation
*Location: `app/generation/`*

Once the most relevant context is identified, the Generation module handles the synthesis of the final answer.
- **Prompt Engineering**: The retrieved chunks are formatted cleanly into a prompt template alongside the user's original query. The LLM is strictly instructed to *only* use the provided context to answer the question, mitigating hallucinations.

## 5. Citation Validation
*Location: `app/citations/`*

To build trust in the output, the system must prove where it got its information.
- **Grounding**: As the LLM generates its response, it is asked to provide citation markers (e.g., `[1]`). The Citation module maps these markers back to the exact source chunks. 
- **Validation**: Ensures that the generated text is actually supported by the chunk it references, preventing "fake citations."

## 6. Agent Orchestration
*Location: `app/agent/`*

Simple RAG pipelines statically route every query through the exact same flow. Our Agentic System is smarter.
- **Query Planner**: An LLM acts as a router. By analyzing the complexity and intent of the user's question, it independently decides whether to use a fast lexical search, a deep semantic search, or a comprehensive reranked hybrid search. It can also opt to reject out-of-scope or unsafe questions before spending expensive retrieval compute.

## 7. Evaluation
*Location: `app/evaluation/`*

In production, you cannot improve what you cannot measure. 
- **Metrics**: The system implements standard Information Retrieval metrics to gauge the quality of the chunks being fetched:
  - **Recall@K**: Measures if the expected relevant chunk appears in the top K results.
  - **MRR (Mean Reciprocal Rank)**: Measures how high up in the ranking the first relevant chunk appears.
  - **nDCG@K**: Measures the overall quality and ranking order of the retrieved list.
- **Eval Set**: A curated dataset of questions mapped to known `chunk_ids` allows developers to test changes to the retrieval pipeline regression-free.

## 8. Continuous Integration (CI)
*Location: `.github/workflows/ci.yml`*

Automated quality control.
- **GitHub Actions**: Every code push or pull request triggers a workflow that installs dependencies, runs the Pytest unit test suite, and executes the evaluation pipeline (`scripts/run_eval.py`). If the retrieval metrics drop below minimum acceptable thresholds, the build fails, preventing regressions from reaching production.

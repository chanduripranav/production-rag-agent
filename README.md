# 🚀 Agentic RAG System for Production

Welcome to the **Agentic RAG System**, a robust, production-ready implementation of Retrieval-Augmented Generation (RAG). Built with advanced retrieval strategies and intelligent agent orchestration, this project is designed to accurately answer complex queries over custom documents while providing transparent, validated citations.

This system is perfect for those looking to understand or deploy a state-of-the-art RAG architecture capable of reasoning about *how* to find information, not just blindly searching.

---

## ✨ Features

- **Hybrid Retrieval Strategy**: Combines the precision of exact keyword matches (BM25) with the deep semantic understanding of vector embeddings (FAISS/Sentence-Transformers) using Reciprocal Rank Fusion (RRF).
- **Intelligent Reranking**: Uses a powerful Cross-Encoder to re-evaluate and sort the top retrieved chunks, dramatically improving the relevance of context sent to the LLM.
- **Agentic Orchestration**: Features a dynamic query planner that analyzes the user's question to decide the optimal retrieval approach, ensuring high accuracy and efficiency.
- **Citation Validation**: Synthesizes answers using a generative LLM while deeply validating that every claim can be directly traced back to specific chunks from the source documents.
- **Evaluation Pipeline**: Built-in scripts to evaluate the quality of retrieval using standard metrics (Recall@K, MRR, nDCG@K).
- **Production-Ready Structure**: Clean, modular codebase following best practices, complete with a FastAPI interface and GitHub Actions CI.

---

## 🏗️ Architecture Flow

```
User Query ──> [ FastAPI ] ──> [ Agent Planner ]
                                     │
                                     ▼
                        (Decides Retrieval Strategy)
                        [BM25]  [Vector]  [Hybrid]
                                     │
                                     ▼
                            [ Cross-Encoder ]
                              (Reranking)
                                     │
                                     ▼
                            [ LLM Generator ] ──> Synthesis
                                     │
                                     ▼
                        [ Citation Validator ] ──> Fact-Checking
                                     │
                                     ▼
                                Final Output
```
*(For a deep dive into the technical design, see [ARCHITECTURE.md](./ARCHITECTURE.md))*

---

## 📁 Folder Structure

```
production-rag-agent/
├── app/
│   ├── api/          # FastAPI application and routing
│   ├── agent/        # Agent orchestration and query planning
│   ├── citations/    # Grounding and citation validation logic
│   ├── evaluation/   # Core evaluation metrics (Recall, MRR, nDCG)
│   ├── generation/   # LLM prompt construction and response synthesis
│   ├── ingestion/    # Document parsing and chunking
│   ├── reranking/    # Cross-Encoder reranking models
│   ├── retrieval/    # BM25, Vector, and Hybrid search implementations
│   └── utils/        # Shared configuration and helpers
├── data/
│   ├── raw/          # Place original PDFs and Word documents here
│   ├── processed/    # Generated chunks and vector indices
│   └── eval/         # Evaluation datasets
├── scripts/          # Executable scripts to run ingestion, retrieval, and evaluation
├── tests/            # Pytest test suite
├── package.json      # Node dependencies (if using frontend)
├── requirements.txt  # Python dependencies
├── pytest.ini        # Testing configuration
├── ARCHITECTURE.md   # Deep dive into system design
└── README.md         # You are here!
```

---

## 🚀 Setup Instructions

This project requires Python 3.10+. Read below to get your local environment running.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/production-rag-agent.git
   cd production-rag-agent
   ```

2. **Set up a Virtual Environment**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   # On MacOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   Place your raw documents (PDFs, DOCX) into the `data/raw/` directory.

---

## 🛠️ How to Use

### 1. Ingestion (Preparing the Data)
Before you can query, you must process your raw documents into searchable indices.
```bash
python scripts/ingest.py
```
*This extracts text, chunks it, creates embeddings, and builds the BM25 and Vector indices in `data/processed/`.*

### 2. Testing Retrieval Subsystems
You can individually test the different retrieval methods to see how they perform.
```bash
# Test basic BM25
python scripts/run_retrieval.py --mode bm25 --query "What is 3D parallax?"

# Test Vector Search
python scripts/run_retrieval.py --mode vector --query "What is 3D parallax?"

# Test Hybrid Search (BM25 + Vector)
python scripts/run_retrieval.py --mode hybrid --query "What is 3D parallax?"

# Test Reranked Hybrid Search (Best retrieval quality)
python scripts/run_retrieval.py --mode rerank --query "What is 3D parallax?"
```

### 3. Running the Full Agentic Flow
To let the Agent Planner dynamically handle the query, retrieve relevant chunks, and generate a synthesized, cited answer:
```bash
python scripts/run_agent.py --query "Explain the difference between geometric primitives and transformations."
```

### 4. Starting the API (FastAPI)
Run the application as a highly-concurrent web service using Uvicorn.
```bash
uvicorn app.api.main:app --reload
```
*You can interact with the API documentation by navigating to `http://localhost:8000/docs` in your browser.*

### 5. Running the Web UI (Streamlit)
To interact with the system via a clean, beginner-friendly web interface:
1. Install UI dependencies: `pip install -r requirements-ui.txt`
2. Start the backend: `uvicorn app.api.main:app --reload`
3. In a new terminal, run the UI: `streamlit run ui/app.py`
4. Open the displayed local URL (usually `http://localhost:8501`) in your browser to ask questions and view cited answers.

### 6. Running the Evaluation Pipeline
Ensure that changes to the retrieval logic don't degrade performance by running the built-in evaluation suite.
```bash
python scripts/run_eval.py
```

---

## 💡 Example Request & Response

When querying the `/chat` endpoint via the API, the Agentic RAG system provides clear, cited responses.

**Request:**
```json
POST /chat
{
  "query": "What is the Aperture Problem?"
}
```

**Response:**
```json
{
  "answer": "The aperture problem refers to the fact that motion estimation is highly ambiguous when the observation window is very small [1].",
  "citations": [
    {
      "chunk_id": "CV UNIT 4 & 5 QB.pdf-p1-c0000",
      "text": "The refers to the fact that motion estimation is highly ambiguous when the observation window is very small. b. aperture problem"
    }
  ]
}
```

---

## 🔮 Future Improvements

While this is a robust foundational implementation, the following enhancements are planned for the future:
- **Frontend UI**: A modern React/Next.js interface for interacting with the agent.
- **Advanced GraphRAG**: Extracting entities and relationships from the text to populate a Knowledge Graph, enabling complex multi-hop reasoning.
- **Streaming Responses**: Token-by-token streaming of the LLM generation for lower latency user experiences.
- **Multi-Modal Support**: Processing images and tables natively alongside text.

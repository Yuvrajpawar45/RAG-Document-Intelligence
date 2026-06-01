<div align="center">

<!-- Animated Banner SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=3000&pause=1000&color=C8701A&center=true&vCenter=true&multiline=true&repeat=false&width=600&height=60&lines=Retrieval+Augmented+Generation+%7C+FastAPI+%7C+FAISS+%7C+Groq+Llama+3" alt="Typing SVG" />

<br/>

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   📖  D o c M i n d                                     ║
║       RAG Document Intelligence                          ║
║                                                          ║
║   Ask anything. Get cited answers. Powered by Groq.     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

<br/>

[![Python](https://img.shields.io/badge/Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-2f6fdb?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=for-the-badge)](https://groq.com)

[![Tests](https://img.shields.io/badge/Tests-16%2F16_Passing-2d7a3a?style=flat-square&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-c8701a?style=flat-square)]()
[![Free](https://img.shields.io/badge/LLM_Cost-100%25_Free-brightgreen?style=flat-square)]()

<br/>

**[🚀 Quick Start](#-quick-start) · [🏗 Architecture](#-architecture) · [🔌 API Reference](#-api-reference) · [💬 Interview Notes](#-interview-talking-points)**

</div>

---

## 📌 What Is DocMind?

DocMind is a **production-grade RAG (Retrieval-Augmented Generation)** application. Upload any PDF or paste raw text — DocMind indexes it, retrieves the most relevant context for your question, and generates a cited answer using **Groq's free Llama 3.3 70B** API.

No cloud vector database. No paid LLM. Fully local-first.

```
You ask:    "What are the key findings in this paper?"
DocMind:    Retrieves top-5 relevant chunks → Sends to Groq → Returns cited answer
            [Source: paper.pdf, Chunk 12] [Source: paper.pdf, Chunk 47]
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **PDF Ingestion** | Extract and index any PDF via PyMuPDF |
| ✏️ **Text Ingestion** | Paste raw text directly from the UI |
| 🔍 **Semantic Search** | `all-MiniLM-L6-v2` embeddings + FAISS IndexFlatIP |
| 🤖 **LLM Answers** | Groq Llama 3.3 70B — free tier, fast inference |
| 📎 **Source Citations** | Every answer cites exact chunks and file names |
| 💬 **Multi-turn Chat** | Last 6 conversation turns passed as context |
| 🧪 **Unit Tested** | 16 tests covering chunking, retrieval, and scoring |
| ⚡ **Local-first** | FAISS index persisted to disk — no external DB |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOCMIND PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐    PDF / Text    ┌─────────────────────────────┐
  │          │ ───────────────► │  1. CHUNKING                │
  │ Streamlit│                  │     500 chars, 100 overlap  │
  │   UI     │                  └──────────────┬──────────────┘
  │          │                                 │
  │  :8501   │                  ┌──────────────▼──────────────┐
  └────┬─────┘                  │  2. EMBEDDING               │
       │                        │     all-MiniLM-L6-v2 (384d) │
       │  HTTP                  │     L2-normalized vectors   │
       │                        └──────────────┬──────────────┘
  ┌────▼─────┐                                 │
  │ FastAPI  │                  ┌──────────────▼──────────────┐
  │ Backend  │                  │  3. FAISS IndexFlatIP       │
  │          │                  │     Cosine similarity       │
  │  :8000   │                  │     Persisted to disk       │
  └──────────┘                  └──────────────┬──────────────┘
                                               │
              User Query                       │  Top-K Chunks
        ──────────────────►   ┌────────────────▼────────────┐
                              │  4. GROQ LLM (Llama 3.3 70B)│
        ◄──────────────────   │     + chat history context  │
              Answer +        └─────────────────────────────┘
              Citations
```

### Why These Choices?

| Decision | Why |
|---|---|
| `IndexFlatIP` over `IndexFlatL2` | Correct cosine similarity when embeddings are L2-normalized. L2 distance gives meaningless scores on normalized vectors. |
| `all-MiniLM-L6-v2` | Best speed/quality tradeoff for CPU. 384-dim, 80MB, runs locally. |
| Character chunking | No extra dependencies. Token-based chunking is noted as a future improvement. |
| FAISS over Chroma/Pinecone | Zero setup, zero cost, fully local. Scales to millions of vectors on one machine. |
| Groq over OpenAI | Llama 3.3 70B on Groq is free tier — no credit card needed for demos. |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Free Groq API Key](https://console.groq.com)

### 1 · Clone

```bash
git clone https://github.com/Yuvrajpawar45/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
```

### 2 · Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3 · Install

```bash
pip install -r requirements.txt
```

### 4 · Configure

```bash
cp .env.example .env
# Open .env and add your Groq key:
# GROQ_API_KEY=your_key_here
# DOCMIND_API_URL=http://localhost:8000
```

### 5 · Run

Open **two terminals**:

```bash
# Terminal 1 — Backend
uvicorn backend.api:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run app.py
```

### 6 · Open

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |

---

## 🗂 Project Structure

```
RAG-Document-Intelligence/
│
├── app.py                  ← Streamlit frontend
├── requirements.txt
├── pytest.ini
├── .env.example
│
├── backend/
│   ├── __init__.py
│   ├── api.py              ← FastAPI routes
│   └── rag_engine.py       ← Core RAG logic
│
├── tests/
│   ├── __init__.py
│   └── test_rag_engine.py  ← 16 unit tests
│
└── data/                   ← Auto-generated
    ├── faiss.index
    └── metadata.pkl
```

---

## 🔌 API Reference

### `GET /health`
Returns `{"status": "ok"}`

### `GET /stats`
```json
{
  "total_chunks": 118,
  "total_documents": 2,
  "sources": ["report.pdf", "notes.txt"]
}
```

### `POST /ingest/pdf`
```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@document.pdf"
```

### `POST /ingest/text`
```json
{
  "text": "Your raw text content here...",
  "source": "my_notes"
}
```

### `POST /query`
```json
{
  "query": "What are the key findings?",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "The key findings are... [Source: report.pdf, Chunk 12]",
  "sources": ["report.pdf"],
  "chunks_used": 5
}
```

### `DELETE /clear`
Wipes the FAISS index and all metadata.

---

## 🧪 Tests

```bash
pytest
```

```
tests/test_rag_engine.py::TestChunking::test_short_text_returns_one_chunk     PASSED
tests/test_rag_engine.py::TestChunking::test_long_text_produces_multiple_chunks PASSED
tests/test_rag_engine.py::TestChunking::test_chunk_ids_are_sequential          PASSED
tests/test_rag_engine.py::TestChunking::test_chunk_source_is_set               PASSED
tests/test_rag_engine.py::TestChunking::test_very_short_text_under_50_chars_skipped PASSED
tests/test_rag_engine.py::TestChunking::test_overlap_means_chunks_share_content PASSED
tests/test_rag_engine.py::TestChunking::test_empty_string_returns_no_chunks    PASSED
tests/test_rag_engine.py::TestIndex::test_empty_index_retrieve_returns_empty   PASSED
tests/test_rag_engine.py::TestIndex::test_ingest_text_adds_chunks              PASSED
tests/test_rag_engine.py::TestIndex::test_retrieve_after_ingest_returns_results PASSED
tests/test_rag_engine.py::TestIndex::test_scores_are_valid_cosine_similarity   PASSED
tests/test_rag_engine.py::TestIndex::test_clear_empties_index                  PASSED
tests/test_rag_engine.py::TestIndex::test_get_stats_reflects_ingested_docs     PASSED
tests/test_rag_engine.py::TestIndex::test_ingest_too_short_returns_zero        PASSED
tests/test_rag_engine.py::TestAnswer::test_answer_with_empty_index_returns_warning PASSED
tests/test_rag_engine.py::TestAnswer::test_answer_returns_expected_keys        PASSED

16 passed in 0.55s
```

---

## 💬 Interview Talking Points

**Q: Why FAISS over a managed vector DB like Pinecone?**
> FAISS runs entirely locally with zero setup and zero cost. For a portfolio project demoing RAG concepts, eliminating infrastructure dependencies makes it easier to run and evaluate. For production at scale, I'd migrate to Pinecone or Weaviate and add metadata filtering.

**Q: Why `IndexFlatIP` instead of `IndexFlatL2`?**
> When embeddings are L2-normalized — which `sentence-transformers` does via `normalize_embeddings=True` — inner product equals cosine similarity. `IndexFlatL2` on normalized vectors gives squared distances that don't map to a valid similarity range. The original code had `IndexFlatL2` with a `1 - dist` score formula, which was a silent correctness bug I identified and fixed.

**Q: How would you scale this to 10 million documents?**
> Replace `IndexFlatIP` with `IndexIVFFlat` or `IndexHNSWFlat` for approximate nearest neighbor search. Add a metadata database (PostgreSQL) alongside FAISS for filtering. Shard the index across multiple nodes. Add async ingestion via a task queue (Celery/Redis).

**Q: What's the biggest limitation of character-based chunking?**
> LLM context windows are measured in tokens, not characters. A 500-character chunk is roughly 125 tokens — quite small. Token-aware chunking (using `tiktoken`) would give more precise context utilization and avoid cutting mid-sentence more reliably.

**Q: How would you evaluate retrieval quality?**
> Use RAGAS — it measures faithfulness (does the answer match retrieved context?), answer relevancy, and context precision/recall without needing labeled datasets. I'd also add a hybrid search layer (BM25 + dense retrieval) and a reranker (cross-encoder) to improve top-k quality.

---

## 🛣 Roadmap

- [ ] Token-based chunking with `tiktoken`
- [ ] Hybrid search (BM25 + dense retrieval)
- [ ] Cross-encoder reranking
- [ ] RAGAS evaluation pipeline
- [ ] Docker + docker-compose setup
- [ ] Multi-user session isolation
- [ ] Cloud vector DB option (Pinecone / Weaviate)

---

## 👤 Author

<div align="center">

**Yuvraj Pawar**

[![GitHub](https://img.shields.io/badge/GitHub-Yuvrajpawar45-181717?style=for-the-badge&logo=github)](https://github.com/Yuvrajpawar45)

*Final-year CS student · AI/ML Engineering*

</div>

---

<div align="center">

```
Built with curiosity. Debugged with patience. Shipped with tests.
```

⭐ If this project helped you understand RAG, consider starring it.

</div>
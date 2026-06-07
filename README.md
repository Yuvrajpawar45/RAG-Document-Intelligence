# DocMind: RAG Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-2f6fdb?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=for-the-badge)](https://groq.com)

Ask questions about PDFs or pasted text and get cited answers using FastAPI, Streamlit, FAISS, sentence-transformers, and Groq.

[Quick Start](#quick-start) | [Chunking Strategies](#chunking-strategies) | [Retrieval Evaluation](#retrieval-evaluation) | [Architecture](#architecture) | [API Reference](#api-reference) | [Deploy](#deploy-to-streamlit-cloud)

---

## Overview

DocMind is a local-first Retrieval-Augmented Generation (RAG) application. Upload a PDF or paste text, then ask questions — DocMind retrieves the most relevant passages and generates grounded, cited answers via Groq's Llama 3.3 70B.

**What makes this different from a basic RAG tutorial:**

- Two chunking strategies with a measurable performance comparison (Recall@3, Recall@5)
- A reproducible retrieval benchmark (`eval/`) — no RAGAS, no OpenAI key needed
- Standalone mode for Streamlit Cloud deployment (no backend required)
- FastAPI backend with a full test suite

---

## Chunking Strategies

DocMind implements and benchmarks two chunking approaches. You can select the strategy per-ingest from the sidebar.

### Fixed chunking (original)

Slides a 500-character window over text with 50-character overlap. Fast and simple — but can cut sentences mid-way, producing truncated embeddings that miss context.

### Sentence-aware chunking (new ✨)

Groups complete sentences into chunks near the target length (500 chars), with a 1-sentence semantic overlap between consecutive chunks.

**Why sentence-aware is better:**

- Embeddings represent complete thoughts, not truncated fragments
- Overlap is semantic (whole sentences) not an arbitrary character count
- Consistently higher recall on factual Q&A benchmarks

### Comparison

| Metric | Fixed (500 chars) | Sentence-aware |
|---|---|---|
| Total chunks (benchmark corpus) | 17 | 14 |
| Avg chunk length | 487 chars | 523 chars |
| **Recall@3** | **70%** | **80%** |
| **Recall@5** | **80%** | **90%** |

> Sentence-aware chunking achieves **+10% Recall@3** on the DocMind benchmark.
> Run `python eval/run_eval.py` to reproduce.

---

## Retrieval Evaluation

DocMind includes a reproducible retrieval benchmark — no RAGAS, no LLM-as-judge, no OpenAI key needed.

### Methodology

- **Corpus**: a self-contained 1,800-word text about RAG systems (`eval/benchmark.json`)
- **Questions**: 10 factual Q&A pairs with ground-truth keyword sets
- **Metric**: Recall@k — a question is a "hit" at k if any of the top-k retrieved chunks contains all ground-truth keywords for that question
- **Index**: fresh in-memory FAISS `IndexFlatIP` per strategy (no data leakage)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, CPU, no API cost)

### Results

| # | Question | Fixed @3 | Sent @3 | Fixed @5 | Sent @5 |
|---|---|---|---|---|---|
| 1 | What is Retrieval-Augmented Generation? | ✅ | ✅ | ✅ | ✅ |
| 2 | Who introduced RAG and when? | ✅ | ✅ | ✅ | ✅ |
| 3 | What embedding model does DocMind use? | ✅ | ✅ | ✅ | ✅ |
| 4 | Why is sentence-aware chunking better? | ❌ | ✅ | ✅ | ✅ |
| 5 | What is Recall@k and why Recall@3? | ✅ | ✅ | ✅ | ✅ |
| 6 | What FAISS index type does DocMind use? | ✅ | ✅ | ✅ | ✅ |
| 7 | How does DocMind prevent hallucination? | ❌ | ✅ | ❌ | ✅ |
| 8 | Streamlit Cloud deployment constraint? | ❌ | ✅ | ✅ | ✅ |
| 9 | What does RAGAS measure? | ✅ | ✅ | ✅ | ✅ |
| 10 | What vector databases can RAG use? | ✅ | ✅ | ✅ | ✅ |
| | **Recall@k** | **70%** | **80%** | **80%** | **90%** |

### Run the benchmark yourself

```bash
# Basic — prints summary table
python eval/run_eval.py

# Verbose — shows per-question hit/miss detail
python eval/run_eval.py --verbose

# Save results to eval/eval_results.md
python eval/run_eval.py --save
```

No API keys needed. The benchmark is entirely local (embedding + FAISS only).

---

## Features

| Feature | Description |
|---|---|
| PDF ingestion | Extracts text from PDF files using PyMuPDF |
| Text ingestion | Paste raw text directly from the UI |
| **Two chunking strategies** | Fixed (500 chars) or Sentence-aware — selectable per-ingest |
| Semantic retrieval | `all-MiniLM-L6-v2` embeddings + FAISS `IndexFlatIP` |
| Cosine similarity | Normalised embeddings with `IndexFlatIP` (exact cosine) |
| LLM answers | Groq Llama 3.3 70B with temperature 0.1 |
| Source citations | Returns cited chunks and source filenames |
| Chat history | Sends recent conversation context to the backend |
| Local persistence | Saves FAISS index and metadata under `data/` |
| **Standalone mode** | Full RAG in-process for Streamlit Cloud (no backend needed) |
| **Retrieval benchmark** | Reproducible Recall@3/5 evaluation — no API keys needed |
| Tests | pytest suite covering chunking, retrieval, scoring, and answers |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS CPU (`IndexFlatIP`) |
| PDF parsing | PyMuPDF |
| LLM provider | Groq (Llama 3.3 70B) |
| Evaluation | Custom Recall@k benchmark (no RAGAS needed) |
| Testing | pytest |
| Configuration | python-dotenv |

---

## Architecture

```
                  +----------------------+
                  |    Streamlit UI      |
                  |      port 8501       |
                  +----------+-----------+
                             |
                   ┌─────────┴─────────┐
                   │ strategy selector │
                   │  fixed | sentence │
                   └─────────┬─────────┘
                             | HTTP  (or direct call in standalone mode)
                             v
                  +----------------------+
                  |    FastAPI Backend   |
                  |      port 8000       |
                  +----------+-----------+
                             |
           +-----------------+-----------------+
           |                                   |
           v                                   v
+----------------------+           +----------------------+
|  PDF/Text Ingestion  |           |     Query Route      |
+----------+-----------+           +----------+-----------+
           |                                  |
    ┌──────┴──────┐                           v
    │  chunk_text │                +----------------------+
    │  (strategy) │                |  Embed user query    |
    └──────┬──────┘                +----------+-----------+
           |                                  |
           v                                  v
+----------------------+           +----------------------+
| Generate embeddings  |           |  Retrieve top chunks |
| all-MiniLM-L6-v2     |           |      from FAISS      |
+----------+-----------+           +----------+-----------+
           |                                  |
           v                                  v
+----------------------+           +----------------------+
|   FAISS IndexFlatIP  |           |   Groq Llama 3.3     |
|   (cosine similarity)|           |   cited answer gen   |
+----------------------+           +----------------------+
```

---

## Project Structure

```
RAG-Document-Intelligence/
│
├── app.py                    # Streamlit frontend (local + standalone mode)
├── requirements.txt          # Python dependencies
├── pytest.ini                # pytest configuration
├── .env.example              # Environment variable template
├── .gitignore
│
├── .streamlit/
│   └── config.toml           # Theme + server config for Streamlit Cloud
│
├── backend/
│   ├── __init__.py
│   ├── api.py                # FastAPI routes (strategy param on ingest)
│   └── rag_engine.py         # Core RAG logic (fixed + sentence chunking)
│
├── eval/
│   ├── benchmark.json        # 10 Q&A pairs + self-contained corpus
│   ├── run_eval.py           # Recall@3/5 comparison script (no API keys)
│   └── eval_results.md       # Pre-computed results (auto-generated)
│
├── tests/
│   ├── __init__.py
│   └── test_rag_engine.py    # Unit tests
│
└── data/                     # Auto-generated local index files
    ├── faiss.index
    └── metadata.pkl
```

---

## Quick Start

### Prerequisites

- Python 3.10 or newer
- A free Groq API key from https://console.groq.com

### 1. Clone the repository

```bash
git clone https://github.com/Yuvrajpawar45/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env # Windows
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
DOCMIND_API_URL=http://localhost:8000
```

### 5. Run the backend

```bash
uvicorn backend.api:app --reload --port 8000
```

- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

### 6. Run the frontend

```bash
streamlit run app.py
```

Frontend: http://localhost:8501

---

## Deploy to Streamlit Cloud

DocMind has a **standalone mode** — the RAG engine runs directly inside the Streamlit process with an in-memory FAISS index. No backend needed.

### Steps

1. Push your repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_key_here"
DOCMIND_STANDALONE = "true"
PERSIST_INDEX = "false"
```

5. Click **Deploy** — you get a public `*.streamlit.app` URL in ~2 minutes.

> **Note:** Streamlit Cloud's filesystem is ephemeral. `PERSIST_INDEX=false` keeps the FAISS index in memory only (per session), which is correct for a public demo.

---

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /stats`

```json
{
  "total_chunks": 118,
  "total_documents": 2,
  "sources": ["report.pdf", "notes.txt"]
}
```

### `POST /ingest/pdf?strategy=sentence`

Upload and index a PDF. Optional `strategy` query param: `fixed` (default) or `sentence`.

```bash
curl -X POST "http://localhost:8000/ingest/pdf?strategy=sentence" \
  -F "file=@document.pdf"
```

Response:

```json
{
  "message": "Ingested 'document.pdf' - 12 chunks",
  "chunks_added": 12,
  "strategy": "sentence",
  "stats": { "total_chunks": 12, "total_documents": 1, "sources": ["document.pdf"] }
}
```

### `POST /ingest/text?strategy=sentence`

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
  "top_k": 5,
  "chat_history": []
}
```

### `DELETE /clear`

```bash
curl -X DELETE http://localhost:8000/clear
```

---

## Testing

```bash
pytest
```

Covers: text chunking (both strategies), chunk overlap, empty/short inputs, ingestion, retrieval, cosine score range, index clearing, stats, answer shape.

---

## API Reference: Chunking Strategies

| Parameter | Value | Description |
|---|---|---|
| `strategy` | `fixed` | 500-char window, 50-char overlap (default) |
| `strategy` | `sentence` | Sentence-boundary grouping, 1-sentence overlap |

---

## Roadmap

- [x] Fixed-size chunking
- [x] Sentence-aware chunking
- [x] Retrieval evaluation benchmark (Recall@3, Recall@5)
- [x] Standalone mode for Streamlit Cloud
- [ ] Token-aware chunking (tiktoken)
- [ ] Hybrid BM25 + dense retrieval
- [ ] Cross-encoder reranking
- [ ] Docker + docker-compose setup
- [ ] Multi-user session isolation

---

## Author

**Yuvraj Pawar**

- GitHub: https://github.com/Yuvrajpawar45
- Project: https://github.com/Yuvrajpawar45/RAG-Document-Intelligence

---

Built as a practical RAG portfolio project — with local retrieval, two benchmarked chunking strategies, a reproducible evaluation pipeline, and Streamlit Cloud deployment.
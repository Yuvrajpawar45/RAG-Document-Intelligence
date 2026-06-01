# DocMind: RAG Document Intelligence

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-2f6fdb?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=for-the-badge)](https://groq.com)

Ask questions about PDFs or pasted text and get cited answers using FastAPI, Streamlit, FAISS, sentence-transformers, and Groq.

[Quick Start](#quick-start) | [Architecture](#architecture) | [API Reference](#api-reference) | [Testing](#testing) | [Interview Notes](#interview-notes)

</div>

---

## Overview

DocMind is a local-first Retrieval-Augmented Generation (RAG) application. It lets you upload a PDF or paste text, indexes the content into semantic chunks, retrieves the most relevant passages for a user question, and sends that context to Groq's Llama 3.3 70B model to generate a grounded answer with source citations.

The project is designed as a clear portfolio-ready RAG system:

- Streamlit frontend for document upload, text ingestion, chat, and index controls.
- FastAPI backend for ingestion, retrieval, querying, stats, and clearing the index.
- FAISS local vector index, persisted to disk.
- `all-MiniLM-L6-v2` embeddings from `sentence-transformers`.
- Groq Llama 3.3 70B for answer generation.
- Unit tests for chunking, retrieval, scoring, stats, and answer behavior.

## Demo Flow

```text
User uploads PDF or pastes text
        |
        v
DocMind chunks the content
        |
        v
Embeddings are generated locally
        |
        v
FAISS stores normalized vectors
        |
        v
User asks a question
        |
        v
Top relevant chunks are retrieved
        |
        v
Groq generates an answer with citations
```

Example:

```text
Question:
What are the key findings in this document?

Answer:
The document highlights three key findings...

Sources:
- report.pdf, Chunk 12
- report.pdf, Chunk 47
```

## Features

| Feature | Description |
| --- | --- |
| PDF ingestion | Extracts text from PDF files using PyMuPDF. |
| Text ingestion | Lets users paste raw text directly from the UI. |
| Semantic retrieval | Uses `all-MiniLM-L6-v2` embeddings and FAISS. |
| Cosine similarity | Uses normalized embeddings with `IndexFlatIP`. |
| LLM answers | Generates responses through Groq's Llama 3.3 70B model. |
| Source citations | Returns cited chunks and source filenames. |
| Chat history | Sends recent conversation context to the backend. |
| Local persistence | Saves FAISS index and metadata under `data/`. |
| Tests | Includes focused unit tests for core RAG behavior. |

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS CPU |
| PDF parsing | PyMuPDF |
| LLM provider | Groq |
| Testing | pytest |
| Configuration | python-dotenv |

## Architecture

```text
                     +----------------------+
                     |    Streamlit UI      |
                     |      port 8501       |
                     +----------+-----------+
                                |
                                | HTTP
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
              v                                  v
   +----------------------+           +----------------------+
   |       Chunking       |           |  Embed user query    |
   |  500 chars, overlap  |           +----------+-----------+
   +----------+-----------+                      |
              |                                  v
              v                       +----------------------+
   +----------------------+           |  Retrieve top chunks |
   | Generate embeddings  |           |      from FAISS      |
   +----------+-----------+           +----------+-----------+
              |                                  |
              v                                  v
   +----------------------+           +----------------------+
   |   FAISS IndexFlatIP  |           |   Groq Llama 3.3     |
   |   data/faiss.index   |           |   answer generation  |
   +----------------------+           +----------------------+
```

## Project Structure

```text
RAG-Document-Intelligence/
|
|-- app.py                    # Streamlit frontend
|-- requirements.txt          # Python dependencies
|-- pytest.ini                # pytest configuration
|-- .env.example              # Environment variable template
|-- .gitignore
|
|-- backend/
|   |-- __init__.py
|   |-- api.py                # FastAPI routes
|   |-- rag_engine.py         # Core RAG logic
|
|-- tests/
|   |-- __init__.py
|   |-- test_rag_engine.py    # Unit tests
|
|-- data/                     # Auto-generated local index files
|   |-- faiss.index
|   |-- metadata.pkl
```

## Quick Start

### Prerequisites

- Python 3.10 or newer
- A free Groq API key from <https://console.groq.com>
- Git, if you want to clone or push the project

### 1. Clone the Repository

```bash
git clone https://github.com/Yuvrajpawar45/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
```

If you already have the project locally, open the project folder directly.

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file from the example:

Windows:

```powershell
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
DOCMIND_API_URL=http://localhost:8000
```

Do not commit your real `.env` file. It should stay private.

### 5. Run the Backend

Open terminal 1:

```bash
uvicorn backend.api:app --reload --port 8000
```

Backend URLs:

- Health check: <http://localhost:8000/health>
- API docs: <http://localhost:8000/docs>

### 6. Run the Frontend

Open terminal 2:

```bash
streamlit run app.py
```

Frontend URL:

- Streamlit app: <http://localhost:8501>

## Usage

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Upload a PDF or paste text into the ingestion panel.
4. Wait for DocMind to index the chunks.
5. Ask questions in the chat box.
6. Review the answer and cited sources.
7. Use the clear index option when you want to reset stored documents.

## API Reference

### `GET /health`

Returns a basic health response.

```json
{
  "status": "ok"
}
```

### `GET /stats`

Returns index statistics.

```json
{
  "total_chunks": 118,
  "total_documents": 2,
  "sources": ["report.pdf", "notes.txt"]
}
```

### `POST /ingest/pdf`

Uploads and indexes a PDF file.

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@document.pdf"
```

Example response:

```json
{
  "message": "Ingested 'document.pdf' - 12 chunks",
  "chunks_added": 12,
  "stats": {
    "total_chunks": 12,
    "total_documents": 1,
    "sources": ["document.pdf"]
  }
}
```

### `POST /ingest/text`

Indexes raw text.

```json
{
  "text": "Your raw text content here...",
  "source": "my_notes"
}
```

### `POST /query`

Asks a question over indexed documents.

```json
{
  "query": "What are the key findings?",
  "chat_history": []
}
```

Example response:

```json
{
  "answer": "The key findings are...",
  "sources": ["report.pdf"],
  "chunks_used": 5,
  "retrieved_chunks": []
}
```

### `DELETE /clear`

Clears the FAISS index and metadata.

```bash
curl -X DELETE http://localhost:8000/clear
```

## Testing

Run all tests:

```bash
pytest
```

The test suite covers:

- Text chunking
- Chunk overlap
- Empty and short input handling
- Text ingestion
- Retrieval behavior
- Cosine similarity score range
- Index clearing
- Stats generation
- Answer response shape

## Key Design Decisions

### Why FAISS?

FAISS keeps the project local-first and simple to run. It avoids cloud vector database setup while still demonstrating real vector search behavior.

### Why `IndexFlatIP`?

The embeddings are L2-normalized. With normalized vectors, inner product is equivalent to cosine similarity. That makes `IndexFlatIP` the correct FAISS index for this setup.

### Why `all-MiniLM-L6-v2`?

It is small, fast on CPU, and strong enough for a portfolio RAG system. Its 384-dimensional embeddings are efficient to store and search.

### Why character-based chunking?

Character chunking keeps the implementation lightweight and dependency-free. A token-aware chunker would be a good future improvement for more precise context control.

### Why Groq?

Groq provides fast hosted inference for open models and works well for demos where latency matters.

## Interview Notes

**Q: What problem does this project solve?**

It solves document question answering by combining retrieval with generation. Instead of asking an LLM to answer from memory, the system retrieves relevant document chunks first and then asks the LLM to answer using that context.

**Q: How would you improve retrieval quality?**

I would add token-aware chunking, hybrid search with BM25 plus dense retrieval, metadata filtering, and a cross-encoder reranker before sending context to the LLM.

**Q: How would you scale it?**

For larger collections, I would move from `IndexFlatIP` to an approximate nearest-neighbor index such as IVF or HNSW, add a metadata database, run ingestion asynchronously, and shard the vector index if needed.

**Q: How would you evaluate it?**

I would measure retrieval precision and recall on a small labeled evaluation set, then use RAG evaluation metrics such as faithfulness, answer relevancy, and context precision.

## Troubleshooting

### `GROQ_API_KEY not found`

Create a `.env` file and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Frontend cannot connect to backend

Make sure the backend is running on port `8000`:

```bash
uvicorn backend.api:app --reload --port 8000
```

Also check that `.env` contains:

```env
DOCMIND_API_URL=http://localhost:8000
```

### `faiss` install error

Make sure you are using a supported Python version and reinstall dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Old documents still appear

Clear the index from the UI or call:

```bash
curl -X DELETE http://localhost:8000/clear
```

## Roadmap

- Token-aware chunking
- Hybrid BM25 plus dense retrieval
- Cross-encoder reranking
- RAG evaluation pipeline
- Docker and docker-compose setup
- Multi-user session isolation
- Optional cloud vector database support

## Author

**Yuvraj Pawar**

- GitHub: <https://github.com/Yuvrajpawar45>
- Project: <https://github.com/Yuvrajpawar45/RAG-Document-Intelligence>

---

Built as a practical RAG portfolio project with local retrieval, cited answers, and test coverage.

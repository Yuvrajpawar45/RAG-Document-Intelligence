# DocMind - RAG Document Intelligence

<div align="center">

![DocMind Banner](https://img.shields.io/badge/DocMind-RAG_Document_Intelligence-c8701a?style=for-the-badge&logo=readthedocs&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Local_Vector_Search-2f6fdb?style=flat-square&logo=databricks&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-5b4b8a?style=flat-square)](https://www.sbert.net)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=flat-square)](https://groq.com)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Parsing-1f2937?style=flat-square)](https://pymupdf.readthedocs.io)

**A document question-answering app that indexes PDFs or pasted text, retrieves relevant chunks with FAISS, and generates cited answers with Groq.**

[Features](#-features) | [Architecture](#-system-architecture) | [Tech Stack](#-tech-stack) | [Getting Started](#-getting-started) | [Usage](#-usage)

</div>

---

## Overview

DocMind is a simple Retrieval-Augmented Generation application for working with documents in a chat-like interface:

- Upload a PDF or paste source text
- Convert content into semantic embeddings
- Store chunks in a local FAISS index
- Retrieve the most relevant chunks for a question
- Generate an answer with source-aware context

The app is split into a FastAPI backend and a Streamlit frontend, which makes it easy to run locally and demo end-to-end.

---

## Features

### Document Ingestion
- Upload PDF files and extract text with PyMuPDF
- Paste raw text directly from the UI
- Chunk long content with overlap for better retrieval quality
- Persist indexed vectors and metadata locally in `data/`

### Question Answering
- Semantic retrieval using `all-MiniLM-L6-v2`
- Top-k similarity search with FAISS
- Answer generation using Groq
- Source citations returned with responses
- Basic multi-turn context support through recent chat history

### Developer Experience
- Clean FastAPI API endpoints for ingest, query, stats, and reset
- Streamlit UI for fast local demos
- Configurable backend URL via `DOCMIND_API_URL`
- Local-first setup with no external vector database required

---

## System Architecture

```text
User Question
    |
    v
+----------------------------------------------+
| DocMind Pipeline                             |
|                                              |
| 1) Ingest PDF or raw text                    |
| 2) Chunk content with overlap                |
| 3) Embed chunks with MiniLM                  |
| 4) Store vectors in local FAISS index        |
| 5) Retrieve top-k chunks for a query         |
| 6) Send retrieved context to Groq            |
| 7) Return answer with citations              |
+----------------------------------------------+
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Document upload, indexing, and chat UI |
| Backend API | FastAPI, Uvicorn | Ingestion, retrieval, query endpoints |
| Embeddings | sentence-transformers | Semantic vector generation |
| Vector Store | FAISS | Local similarity search |
| LLM | Groq | Final answer generation |
| PDF Parsing | PyMuPDF | Extract text from uploaded PDFs |
| Persistence | Pickle + FAISS index files | Save chunks and vectors locally |

---

## Project Structure

```text
RAG-Document-Intelligence/
|
|- app.py
|- requirements.txt
|- README.md
|- .env                  (ignored)
|- .vscode/
|  `- settings.json
|
|- backend/
|  |- __init__.py
|  |- api.py
|  `- rag_engine.py
|
`- data/
   |- faiss.index        (generated)
   `- metadata.pkl       (generated)
```

---

## Getting Started

### Prerequisites
- Python 3.10 recommended
- Groq API key
- Internet access on first model download

### 1) Clone

```bash
git clone https://github.com/Yuvrajpawar45/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
```

### 2) Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
DOCMIND_API_URL=http://localhost:8000
```

If you run the backend on another port, update `DOCMIND_API_URL` to match.

### 5) Run Backend

```bash
uvicorn backend.api:app --reload --port 8000
```

### 6) Run Frontend

```bash
streamlit run app.py
```

### 7) Open the App

- UI: `http://127.0.0.1:8501`
- API docs: `http://127.0.0.1:8000/docs`

---

## Usage

### Index Content
1. Open the Streamlit UI
2. Upload a PDF or paste text into the sidebar
3. Click `Index This PDF` or `Index Text`

### Ask Questions
1. Type a question in the main input box
2. Submit it through the UI
3. Review the generated answer and cited sources

### Sample Text

```text
DocMind is a retrieval-augmented generation demo. It uses sentence-transformers for embeddings, FAISS for vector search, and Groq for final answer generation. The system stores indexed chunks locally and can answer questions with source citations.
```

### Sample Question

```text
What does DocMind use for vector search?
```

---

## API Endpoints

- `GET /health`
- `GET /stats`
- `POST /ingest/pdf`
- `POST /ingest/text`
- `POST /query`
- `DELETE /clear`

---

## Notes

- The first backend startup may download model files from Hugging Face.
- Indexed vectors and metadata are stored locally in the `data/` directory.
- `.env` is intentionally ignored and should never be committed.
- The UI reads the backend base URL from `DOCMIND_API_URL`.

---

## Future Enhancements

- Better citation formatting in answers
- Multi-document source comparison
- Per-user chat/session history
- Dockerized local deployment
- Cloud vector database option
- Authentication and access control

---

## Author

Maintained by **Yuvraj Pawar**  
GitHub: [Yuvrajpawar45](https://github.com/Yuvrajpawar45)

---

## License

Use according to your repository license policy.

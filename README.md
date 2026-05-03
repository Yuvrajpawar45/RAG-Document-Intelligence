# DocMind

DocMind is a Retrieval-Augmented Generation (RAG) app for asking questions over PDFs or pasted text. It uses `sentence-transformers` for embeddings, `FAISS` for vector search, `FastAPI` for the backend, and `Streamlit` for the UI.

## Features

- Upload a PDF and index it locally
- Paste raw text and index it as a source
- Ask natural-language questions over indexed content
- Get answers with source citations
- Keep the vector index on disk in the `data/` folder

## Tech Stack

- Frontend: `Streamlit`
- Backend: `FastAPI` + `uvicorn`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: `FAISS`
- LLM: `Groq`
- PDF parsing: `PyMuPDF`

## Project Structure

```text
rag-final/
|-- app.py
|-- backend/
|   |-- __init__.py
|   |-- api.py
|   `-- rag_engine.py
|-- data/
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10 recommended
- A Groq API key

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from the example:

```powershell
copy .env.example .env
```

4. Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
DOCMIND_API_URL=http://localhost:8000
```

## Run the App

Open two terminals in the project root.

Terminal 1: start the backend

```powershell
venv\Scripts\activate
uvicorn backend.api:app --reload --port 8000
```

Terminal 2: start the UI

```powershell
venv\Scripts\activate
streamlit run app.py
```

Open the UI at:

- `http://127.0.0.1:8501`

API docs are available at:

- `http://127.0.0.1:8000/docs`

## Using a Different Backend Port

If you want to run FastAPI on a different port, update `DOCMIND_API_URL` in `.env`.

Example:

```env
DOCMIND_API_URL=http://localhost:8001
```

Then start the backend on the matching port:

```powershell
uvicorn backend.api:app --reload --port 8001
```

## How to Use

1. Open the Streamlit UI.
2. Upload a PDF or paste text in the sidebar.
3. Click `Index This PDF` or `Index Text`.
4. Ask a question in the main chat area.

Sample text to try:

```text
DocMind is a retrieval-augmented generation demo. It uses sentence-transformers for embeddings, FAISS for vector search, and Groq for final answer generation. The system stores indexed chunks locally and can answer questions with source citations.
```

Sample question:

```text
What does DocMind use for vector search?
```

## Notes

- The first backend startup may download the embedding model from Hugging Face.
- Indexed files and metadata are stored in `data/`.
- If no document has been indexed yet, the app will prompt you to upload or paste content first.

## API Endpoints

- `GET /health`
- `GET /stats`
- `POST /ingest/pdf`
- `POST /ingest/text`
- `POST /query`
- `DELETE /clear`

## GitHub Push Checklist

Before pushing:

- Keep `.env` out of Git
- Make sure `venv/` is ignored
- Commit source files, `requirements.txt`, and `README.md`

Example:

```powershell
git add app.py backend .env.example requirements.txt README.md .vscode/settings.json
git commit -m "Make API URL configurable and refresh project docs"
git push
```

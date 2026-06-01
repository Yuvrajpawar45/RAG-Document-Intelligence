"""
FastAPI Backend — RAG Q&A API
Run: uvicorn backend.api:app --reload --port 8000
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine

app = FastAPI(title="RAG Q&A API", version="2.1.0")

# NOTE: allow_origins=["*"] is fine for local development.
# For production, replace "*" with your actual frontend domain:
# allow_origins=["https://yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PDF_SIZE_MB = 20  # reject files larger than this

try:
    rag = RAGEngine()
except EnvironmentError as e:
    import sys
    print(f"\n❌ Startup failed: {e}\n")
    sys.exit(1)


class TextIngestRequest(BaseModel):
    text: str
    source: str = "manual_input"


class QueryRequest(BaseModel):
    query: str
    chat_history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return rag.get_stats()


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_PDF_SIZE_MB} MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_added = rag.ingest_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "message": f"✅ Ingested '{file.filename}' — {chunks_added} chunks",
        "chunks_added": chunks_added,
        "stats": rag.get_stats()
    }


@app.post("/ingest/text")
def ingest_text(req: TextIngestRequest):
    if len(req.text.strip()) < 50:
        raise HTTPException(400, "Text too short (minimum 50 characters)")
    try:
        chunks_added = rag.ingest_text(req.text, req.source)
    except Exception as e:
        raise HTTPException(500, f"Failed to ingest text: {str(e)}")
    return {
        "message": f"✅ Ingested '{req.source}' — {chunks_added} chunks",
        "chunks_added": chunks_added,
        "stats": rag.get_stats()
    }


@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    try:
        return rag.answer(req.query, req.chat_history)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.delete("/clear")
def clear_index():
    rag.clear_index()
    return {"message": "Index cleared"}

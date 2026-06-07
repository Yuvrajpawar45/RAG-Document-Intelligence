"""
DocMind RAG Engine
==================
Core retrieval-augmented generation logic.

Chunking strategies
-------------------
- "fixed"    : character-based sliding window (original behaviour, default)
- "sentence" : sentence-boundary-aware grouping

Pass strategy="sentence" to ingest_text() / ingest_pdf_bytes() to use the
improved chunker. The FastAPI layer exposes this via an optional ?strategy= param.
"""

from __future__ import annotations

import os
import re
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Literal

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBED_DIM          = 384
GROQ_MODEL         = "llama-3.3-70b-versatile"
TOP_K              = 5

FIXED_CHUNK_SIZE   = 500
FIXED_OVERLAP      = 50
SENT_TARGET_CHARS  = 500
SENT_OVERLAP_SENTS = 1

DATA_DIR   = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH  = DATA_DIR / "metadata.pkl"

ChunkStrategy = Literal["fixed", "sentence"]

# ── Singleton state ─────────────────────────────────────────────────────────────
_model:       Optional[SentenceTransformer] = None
_index:       Optional[faiss.IndexFlatIP]   = None
_metadata:    List[Dict]                    = []
_groq_client: Optional[Groq]               = None

PERSIST = os.getenv("PERSIST_INDEX", "true").lower() == "true"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_index() -> faiss.IndexFlatIP:
    global _index, _metadata
    if _index is None:
        if PERSIST and INDEX_PATH.exists() and META_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                _metadata = pickle.load(f)
        else:
            _index    = faiss.IndexFlatIP(EMBED_DIM)
            _metadata = []
    return _index


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _save_index() -> None:
    if not PERSIST:
        return
    DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(_get_index(), str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)


# ── Fixed chunking (original behaviour) ────────────────────────────────────────
def chunk_text_fixed(
    text:       str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap:    int = FIXED_OVERLAP,
) -> List[str]:
    """
    Slide a fixed character window over *text*.
    Fast and simple; may cut sentences mid-way.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── Sentence-aware chunking (new) ───────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter — no NLTK download needed.
    Splits on . ! ? followed by whitespace, keeping the punctuation.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_text_sentence(
    text:          str,
    target_chars:  int = SENT_TARGET_CHARS,
    overlap_sents: int = SENT_OVERLAP_SENTS,
) -> List[str]:
    """
    Groups complete sentences into chunks near *target_chars*.
    Overlaps *overlap_sents* whole sentences between consecutive chunks
    so cross-boundary context is preserved.

    Why this beats fixed chunking
    --------------------------------
    - Embeddings represent complete thoughts, not truncated fragments.
    - Overlap is semantic (whole sentences) not an arbitrary char count.
    - Recall improves for questions whose answers span a sentence boundary.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    chunks, i = [], 0
    while i < len(sentences):
        group, chars, j = [], 0, i
        while j < len(sentences):
            s = sentences[j]
            if group and chars + len(s) > target_chars:
                break
            group.append(s)
            chars += len(s) + 1
            j += 1
        chunks.append(" ".join(group))
        i += max(1, len(group) - overlap_sents)
    return chunks


# ── Public chunking router ──────────────────────────────────────────────────────
def chunk_text(
    text:          str,
    strategy:      ChunkStrategy = "fixed",
    chunk_size:    int = FIXED_CHUNK_SIZE,
    overlap:       int = FIXED_OVERLAP,
    target_chars:  int = SENT_TARGET_CHARS,
    overlap_sents: int = SENT_OVERLAP_SENTS,
) -> List[str]:
    """Dispatch to the requested chunking strategy."""
    if strategy == "sentence":
        return chunk_text_sentence(text, target_chars, overlap_sents)
    return chunk_text_fixed(text, chunk_size, overlap)


# ── Embedding ───────────────────────────────────────────────────────────────────
def _embed(texts: List[str]) -> np.ndarray:
    vecs = _get_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    return vecs.astype("float32")


# ── Ingestion ───────────────────────────────────────────────────────────────────
def ingest_text(
    text:     str,
    source:   str           = "pasted_text",
    strategy: ChunkStrategy = "fixed",
) -> Dict:
    global _metadata
    index  = _get_index()
    chunks = chunk_text(text, strategy=strategy)
    if not chunks:
        return {"chunks_added": 0, "strategy": strategy}

    vecs  = _embed(chunks)
    index.add(vecs)
    start = len(_metadata)
    for i, chunk in enumerate(chunks):
        _metadata.append({
            "text":     chunk,
            "source":   source,
            "chunk_id": start + i,
            "strategy": strategy,
        })
    _save_index()
    return {"chunks_added": len(chunks), "strategy": strategy, "stats": get_stats()}


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename:  str,
    strategy:  ChunkStrategy = "fixed",
) -> Dict:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        doc       = fitz.open(tmp_path)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    finally:
        os.unlink(tmp_path)

    return ingest_text(full_text, source=filename, strategy=strategy)


# ── Retrieval ───────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    index = _get_index()
    if index.ntotal == 0:
        return []
    q_vec           = _embed([query])
    k               = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta          = _metadata[idx].copy()
        meta["score"] = float(score)
        results.append(meta)
    return results


# ── Generation ──────────────────────────────────────────────────────────────────
def generate_answer(
    query:        str,
    chunks:       List[Dict],
    chat_history: Optional[List[Dict]] = None,
) -> str:
    if not chunks:
        return "I couldn't find relevant information to answer that question."

    context = "\n\n".join(
        f"[{i}] (source: {c['source']}, chunk {c['chunk_id']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    system_prompt = (
        "You are DocMind, a precise document Q&A assistant. "
        "Answer ONLY from the provided context. "
        "Cite sources using [1], [2] etc. "
        "If the context does not contain enough information, say so clearly."
    )
    messages = list((chat_history or [])[-4:])
    messages.append({
        "role":    "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })
    response = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ── Stats & housekeeping ─────────────────────────────────────────────────────────
def get_stats() -> Dict:
    sources = list({m["source"] for m in _metadata})
    return {
        "total_chunks":    len(_metadata),
        "total_documents": len(sources),
        "sources":         sources,
    }


def clear_index() -> None:
    global _index, _metadata
    _index    = faiss.IndexFlatIP(EMBED_DIM)
    _metadata = []
    if PERSIST:
        DATA_DIR.mkdir(exist_ok=True)
        faiss.write_index(_index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(_metadata, f) 
# ── Compatibility shim — api.py uses RAGEngine class + chunk_stats ──────────────

def chunk_stats(text: str, strategy: str = "fixed") -> dict:
    chunks = chunk_text(text, strategy=strategy)
    return {
        "chunks":     len(chunks),
        "avg_chars":  round(sum(len(c) for c in chunks) / max(len(chunks), 1), 1),
        "sample":     chunks[0][:200] if chunks else "",
    }


class RAGEngine:
    """Thin class wrapper so api.py can do rag = RAGEngine()."""

    def ingest_text(self, text: str, source: str = "pasted_text", strategy: str = "fixed") -> int:
        result = ingest_text(text, source=source, strategy=strategy)
        return result["chunks_added"]

    def ingest_pdf(self, pdf_path: str, strategy: str = "fixed") -> int:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        filename = Path(pdf_path).name
        result = ingest_pdf_bytes(pdf_bytes, filename=filename, strategy=strategy)
        return result["chunks_added"]

    def answer(self, query: str, chat_history: list = []) -> dict:
        chunks = retrieve(query)
        answer = generate_answer(query, chunks, chat_history=chat_history)
        return {
            "answer":      answer,
            "sources":     list({c["source"] for c in chunks}),
            "chunks_used": len(chunks),
        }

    def get_stats(self) -> dict:
        return get_stats()

    def clear_index(self) -> None:
        clear_index()
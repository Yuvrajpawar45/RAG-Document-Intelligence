"""
RAG Engine — Core retrieval and generation logic
Embeddings: sentence-transformers (all-MiniLM-L6-v2)
Vector DB:  FAISS (local, no server)
LLM:        Llama 3 via Groq API (FREE)
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF
from nltk.tokenize import PunktSentenceTokenizer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"        # fast, CPU-friendly (384-dim)
GROQ_MODEL    = "llama-3.3-70b-versatile"  # FREE on Groq
CHUNK_SIZE    = 500                         # characters per chunk
                                            # NOTE: token-based chunking (e.g. tiktoken)
                                            # is more precise for LLM context windows,
                                            # but character-based is dependency-free.
CHUNK_OVERLAP = 100                         # overlap to avoid losing context at boundaries
TOP_K         = 5                           # chunks retrieved per query
CHUNK_STRATEGIES = ("fixed", "sentence")
INDEX_PATH    = Path("data/faiss.index")
META_PATH     = Path("data/metadata.pkl")
# ─────────────────────────────────────────────────────────────────────────────


def _fixed_chunks(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:min(start + CHUNK_SIZE, len(text))].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _sentence_chunks(text: str) -> list[str]:
    sentences = PunktSentenceTokenizer().tokenize(text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > CHUNK_SIZE:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(_fixed_chunks(sentence))
            continue

        candidate = " ".join(current + [sentence])
        if current and len(candidate) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                added_length = len(previous) + (1 if overlap else 0)
                if overlap_length + added_length > CHUNK_OVERLAP:
                    break
                overlap.insert(0, previous)
                overlap_length += added_length
            while overlap and len(" ".join(overlap + [sentence])) > CHUNK_SIZE:
                overlap.pop(0)
            current = overlap

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk) > 50]


def chunk_text(text: str, strategy: str = "fixed") -> list[str]:
    """Split text using fixed character windows or sentence-aware windows."""
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Choose from: {', '.join(CHUNK_STRATEGIES)}"
        )

    text = text.strip()
    if not text:
        return []
    return _fixed_chunks(text) if strategy == "fixed" else _sentence_chunks(text)


def chunk_stats(text: str, strategy: str) -> dict:
    chunks = chunk_text(text, strategy)
    lengths = [len(chunk) for chunk in chunks]
    return {
        "strategy": strategy,
        "chunk_count": len(chunks),
        "average_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_chars": min(lengths, default=0),
        "max_chunk_chars": max(lengths, default=0),
    }


class RAGEngine:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Create a .env file with: GROQ_API_KEY=your_key_here\n"
                "Get a free key at: https://console.groq.com"
            )

        logger.info("Loading embedding model: %s", EMBED_MODEL)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.client = Groq(api_key=api_key)

        # NOTE: Using IndexFlatIP (inner product) — correct for cosine similarity
        # when embeddings are L2-normalized (which we do via normalize_embeddings=True).
        # Do NOT use IndexFlatL2 with normalized embeddings; the scores would be meaningless.
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: list[dict] = []

        if INDEX_PATH.exists() and META_PATH.exists():
            self._load_index()
            logger.info("Loaded existing index — %d chunks", len(self.chunks))
        else:
            logger.info("Empty index — ingest a document to get started")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, strategy: str = "fixed") -> int:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        return self._ingest_text(full_text, Path(pdf_path).name, strategy)

    def ingest_text(self, text: str, source: str = "manual_input", strategy: str = "fixed") -> int:
        return self._ingest_text(text, source, strategy)

    def _ingest_text(self, text: str, source: str, strategy: str) -> int:
        chunks = self._chunk_text(text, source, strategy)
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True   # required for cosine similarity via IndexFlatIP
        )
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)
        self._save_index()
        return len(chunks)

    def _chunk_text(self, text: str, source: str, strategy: str = "fixed") -> list[dict]:
        return [
            {
                "text": chunk,
                "source": source,
                "chunk_id": chunk_id,
                "chunking_strategy": strategy,
            }
            for chunk_id, chunk in enumerate(chunk_text(text, strategy))
        ]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_emb = np.array(
            self.embedder.encode([query], normalize_embeddings=True)
        ).astype("float32")
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        # With IndexFlatIP + normalized vectors, score = cosine similarity ∈ [-1, 1]
        # Higher is better (1.0 = identical)
        return [
            {**self.chunks[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx < len(self.chunks)
        ]

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str, chat_history: list[dict] = []) -> dict:
        retrieved = self.retrieve(query)
        if not retrieved:
            return {
                "answer": "⚠️ No documents ingested yet. Please upload a PDF or paste some text first.",
                "sources": [],
                "chunks_used": 0
            }

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']} | Chunk {c['chunk_id']}]\n{c['text']}"
            for c in retrieved
        )

        system_prompt = f"""You are a precise, helpful assistant answering questions strictly from the provided document context.

Rules:
- Answer ONLY from the context. Never use outside knowledge.
- If not found in context, say "I couldn't find that in the provided documents."
- Be concise, clear, and well-structured.
- Always end with citations: [Source: filename, Chunk N]

CONTEXT:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1024,
            messages=messages
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": list({c["source"] for c in retrieved}),
            "chunks_used": len(retrieved),
            "retrieved_chunks": retrieved
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    # NOTE: pickle is used here for local development only.
    # For production, replace with SQLite or a proper database —
    # loading pickle from untrusted sources is a security risk.

    def _save_index(self):
        INDEX_PATH.parent.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load_index(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def clear_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()

    def get_stats(self) -> dict:
        sources = list({c["source"] for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(sources),
            "sources": sources
        }

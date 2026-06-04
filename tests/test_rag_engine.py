"""
Unit tests for RAGEngine core logic.
Run: pytest tests/test_rag_engine.py -v

These tests mock external dependencies (SentenceTransformer, Groq)
so no API key or internet connection is needed.
"""

import os
import sys
import types
import pytest
import numpy as np

# ── Mock external deps before importing rag_engine ───────────────────────────

# Mock sentence_transformers
st_mod = types.ModuleType("sentence_transformers")
class MockEmbedder:
    def __init__(self, model_name=None, *args, **kwargs):
        pass  # accept and ignore the model name string
    def get_sentence_embedding_dimension(self):
        return 384
    def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
        vecs = np.random.rand(len(texts), 384).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-8)
        return vecs
st_mod.SentenceTransformer = MockEmbedder
sys.modules["sentence_transformers"] = st_mod

# Mock groq
groq_mod = types.ModuleType("groq")
class MockGroq:
    def __init__(self, api_key=None): pass
groq_mod.Groq = MockGroq
sys.modules["groq"] = groq_mod

# Mock fitz (PyMuPDF)
fitz_mod = types.ModuleType("fitz")
sys.modules["fitz"] = fitz_mod

# Set env var so RAGEngine doesn't raise on missing key
os.environ["GROQ_API_KEY"] = "test_key_not_real"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.rag_engine import RAGEngine, CHUNK_SIZE, CHUNK_OVERLAP, chunk_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Fresh RAGEngine with tmp data directory."""
    monkeypatch.chdir(tmp_path)
    e = RAGEngine()
    return e


# ── Chunking tests ────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_text_returns_one_chunk(self, engine):
        text = "A" * 200
        chunks = engine._chunk_text(text, "test.pdf")
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self, engine):
        text = "B" * (CHUNK_SIZE * 3)
        chunks = engine._chunk_text(text, "test.pdf")
        assert len(chunks) > 1

    def test_chunk_ids_are_sequential(self, engine):
        text = "C" * (CHUNK_SIZE * 2)
        chunks = engine._chunk_text(text, "test.pdf")
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunk_source_is_set(self, engine):
        chunks = engine._chunk_text("D" * 300, "my_doc.pdf")
        assert all(c["source"] == "my_doc.pdf" for c in chunks)

    def test_very_short_text_under_50_chars_skipped(self, engine):
        chunks = engine._chunk_text("tiny", "test.pdf")
        assert len(chunks) == 0

    def test_overlap_means_chunks_share_content(self, engine):
        word = "hello "
        text = word * 200
        chunks = engine._chunk_text(text, "src")
        if len(chunks) >= 2:
            end_of_first = chunks[0]["text"][-(CHUNK_OVERLAP):]
            start_of_second = chunks[1]["text"][:CHUNK_OVERLAP]
            assert len(set(end_of_first) & set(start_of_second)) > 0

    def test_empty_string_returns_no_chunks(self, engine):
        chunks = engine._chunk_text("", "test.pdf")
        assert chunks == []

    def test_sentence_strategy_keeps_sentence_boundaries(self):
        sentence = "Sentence-aware chunking keeps this complete sentence intact. "
        chunks = chunk_text(sentence * 20, "sentence")
        assert len(chunks) > 1
        assert all(chunk.endswith(".") for chunk in chunks)
        assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("A sufficiently long sample sentence." * 3, "unknown")


# ── Index / retrieval tests ───────────────────────────────────────────────────

class TestIndex:
    def test_empty_index_retrieve_returns_empty(self, engine):
        results = engine.retrieve("anything")
        assert results == []

    def test_ingest_text_adds_chunks(self, engine):
        n = engine.ingest_text("E" * 300, "source_a")
        assert n > 0
        assert engine.index.ntotal == n

    def test_ingest_text_records_sentence_strategy(self, engine):
        engine.ingest_text("One sentence about AI. " * 30, "source_a", "sentence")
        assert all(c["chunking_strategy"] == "sentence" for c in engine.chunks)

    def test_retrieve_after_ingest_returns_results(self, engine):
        engine.ingest_text("The sky is blue and the grass is green. " * 20, "nature.txt")
        results = engine.retrieve("sky")
        assert len(results) > 0

    def test_scores_are_valid_cosine_similarity(self, engine):
        """Scores must be in [-1, 1] — valid cosine similarity range."""
        engine.ingest_text("F" * 500, "test")
        results = engine.retrieve("query")
        for r in results:
            assert -1.0 <= r["score"] <= 1.0, (
                f"Invalid score {r['score']} — check if IndexFlatIP is used with normalized embeddings"
            )

    def test_clear_empties_index(self, engine):
        engine.ingest_text("G" * 300, "source")
        engine.clear_index()
        assert engine.index.ntotal == 0
        assert engine.chunks == []

    def test_get_stats_reflects_ingested_docs(self, engine):
        engine.ingest_text("H" * 300, "doc_a.pdf")
        engine.ingest_text("I" * 300, "doc_b.pdf")
        stats = engine.get_stats()
        assert stats["total_documents"] == 2
        assert "doc_a.pdf" in stats["sources"]
        assert "doc_b.pdf" in stats["sources"]

    def test_ingest_too_short_returns_zero(self, engine):
        n = engine.ingest_text("short", "tiny.txt")
        assert n == 0


# ── Answer tests ──────────────────────────────────────────────────────────────

class TestAnswer:
    def test_answer_with_empty_index_returns_warning(self, engine):
        result = engine.answer("What is this?")
        assert "No documents ingested" in result["answer"]
        assert result["chunks_used"] == 0

    def test_answer_returns_expected_keys(self, engine):
        engine.ingest_text("Python is a programming language. " * 20, "python.txt")

        # Patch the Groq client to avoid real API call
        # Using plain instances (not nested classes) so choices[0] works correctly
        class FakeMessage:
            content = "Mocked answer."

        class FakeChoice:
            message = FakeMessage()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                return FakeCompletion()

        class FakeChat:
            completions = FakeCompletions()

        class FakeGroqClient:
            chat = FakeChat()

        engine.client = FakeGroqClient()

        result = engine.answer("What is Python?")
        assert "answer" in result
        assert "sources" in result
        assert "chunks_used" in result
        assert result["chunks_used"] > 0

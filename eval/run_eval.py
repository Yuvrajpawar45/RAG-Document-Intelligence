"""
DocMind Retrieval Evaluation
=============================
Compares Fixed vs Sentence-aware chunking on a 10-question benchmark.
Computes Recall@3 and Recall@5 for each strategy — no RAGAS, no OpenAI key needed.

Usage
-----
    python eval/run_eval.py              # run both strategies, print table
    python eval/run_eval.py --save       # also write eval/eval_results.md
    python eval/run_eval.py --verbose    # print per-question detail

How it works
------------
1. Load benchmark.json (corpus + 10 Q&A pairs with ground-truth keywords).
2. For each chunking strategy:
   a. Chunk the corpus and build a fresh in-memory FAISS index.
   b. For each question, retrieve top-5 chunks.
   c. A question is a "hit" at k if any of the top-k chunks contains
      ALL of the question's ground_truth_keywords (case-insensitive).
3. Report Recall@3 and Recall@5 side-by-side.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple

# ── Make sure we can import from the project root ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Reuse chunking functions directly ─────────────────────────────────────────
from backend.rag_engine import chunk_text_fixed, chunk_text_sentence

EMBED_MODEL = "all-MiniLM-L6-v2"
BENCHMARK   = Path(__file__).parent / "benchmark.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _embed(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def _build_index(
    corpus: str,
    strategy: str,
    model: SentenceTransformer,
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Chunk corpus, embed, return (index, chunks)."""
    if strategy == "fixed":
        chunks = chunk_text_fixed(corpus)
    else:
        chunks = chunk_text_sentence(corpus)

    vecs  = _embed(model, chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks


def _hit(retrieved_chunks: List[str], keywords: List[str]) -> bool:
    """True if ANY retrieved chunk contains ALL keywords (case-insensitive)."""
    kws_lower = [k.lower() for k in keywords]
    for chunk in retrieved_chunks:
        cl = chunk.lower()
        if all(kw in cl for kw in kws_lower):
            return True
    return False


def evaluate(
    corpus:    str,
    questions: List[Dict],
    strategy:  str,
    model:     SentenceTransformer,
    verbose:   bool = False,
) -> Dict:
    index, chunks = _build_index(corpus, strategy, model)
    results = []

    for q in questions:
        query    = q["question"]
        keywords = q["ground_truth_keywords"]

        q_vec           = _embed(model, [query])
        k               = min(5, index.ntotal)
        scores, indices = index.search(q_vec, k)

        top5 = [chunks[i] for i in indices[0] if i >= 0]
        top3 = top5[:3]

        hit3 = _hit(top3, keywords)
        hit5 = _hit(top5, keywords)

        if verbose:
            status = "✓" if hit3 else ("~" if hit5 else "✗")
            print(f"  [{status}] Q{q['id']:02d}: {query}")
            if not hit3 and top3:
                snippet = top3[0][:120].replace("\n", " ")
                print(f"       top-1: {snippet}…")

        results.append({"id": q["id"], "question": query, "hit@3": hit3, "hit@5": hit5})

    recall3 = sum(r["hit@3"] for r in results) / len(results)
    recall5 = sum(r["hit@5"] for r in results) / len(results)

    chunks_info = chunk_text_fixed(corpus) if strategy == "fixed" else chunk_text_sentence(corpus)
    avg_len = sum(len(c) for c in chunks_info) / max(len(chunks_info), 1)

    return {
        "strategy":        strategy,
        "total_chunks":    len(chunks_info),
        "avg_chunk_chars": round(avg_len, 1),
        "recall@3":        round(recall3, 3),
        "recall@5":        round(recall5, 3),
        "per_question":    results,
    }


# ── Results table ─────────────────────────────────────────────────────────────
def _markdown_table(fixed: Dict, sentence: Dict) -> str:
    rows = [
        ("Strategy",        "Fixed (500 chars)",              "Sentence-aware"),
        ("Total chunks",    str(fixed["total_chunks"]),        str(sentence["total_chunks"])),
        ("Avg chunk length",f"{fixed['avg_chunk_chars']} chars",f"{sentence['avg_chunk_chars']} chars"),
        ("**Recall@3**",    f"**{fixed['recall@3']:.0%}**",    f"**{sentence['recall@3']:.0%}**"),
        ("**Recall@5**",    f"**{fixed['recall@5']:.0%}**",    f"**{sentence['recall@5']:.0%}**"),
    ]
    lines = [
        "| Metric | Fixed chunking | Sentence-aware chunking |",
        "|--------|---------------|------------------------|",
    ]
    for label, v_fixed, v_sent in rows:
        lines.append(f"| {label} | {v_fixed} | {v_sent} |")
    return "\n".join(lines)


def _per_question_table(fixed: Dict, sentence: Dict) -> str:
    lines = [
        "| # | Question | Fixed @3 | Sent @3 | Fixed @5 | Sent @5 |",
        "|---|----------|----------|---------|----------|---------|",
    ]
    for fq, sq in zip(fixed["per_question"], sentence["per_question"]):
        q = fq["question"][:55] + ("…" if len(fq["question"]) > 55 else "")
        f3 = "✅" if fq["hit@3"] else "❌"
        s3 = "✅" if sq["hit@3"] else "❌"
        f5 = "✅" if fq["hit@5"] else "❌"
        s5 = "✅" if sq["hit@5"] else "❌"
        lines.append(f"| {fq['id']:2d} | {q} | {f3} | {s3} | {f5} | {s5} |")
    return "\n".join(lines)


def _save_markdown(fixed: Dict, sentence: Dict, out_path: Path) -> None:
    content = textwrap.dedent(f"""\
        # DocMind — Retrieval Evaluation Results

        Benchmark: 10 questions against a self-contained RAG/ML corpus.
        Metric: **Recall@k** — fraction of questions where the ground-truth
        passage appeared in the top-k retrieved chunks.

        ## Summary

        {_markdown_table(fixed, sentence)}

        > Sentence-aware chunking improves Recall@3 by
        > **{(sentence['recall@3'] - fixed['recall@3']):.0%}** over fixed chunking
        > on this benchmark.

        ## Per-question breakdown

        {_per_question_table(fixed, sentence)}

        ## Methodology

        - Embedding model: `all-MiniLM-L6-v2` (384-dim, CPU, no API key)
        - Vector index: FAISS `IndexFlatIP` with L2-normalised vectors (= cosine similarity)
        - Fixed chunking: 500 chars, 50-char overlap
        - Sentence chunking: target 500 chars per chunk, 1-sentence overlap
        - A question is a "hit" if **any** top-k chunk contains all ground-truth keywords
        - No LLM involved in evaluation — purely retrieval quality
    """)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n📄 Results saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="DocMind retrieval benchmark")
    parser.add_argument("--save",    action="store_true", help="Write eval/eval_results.md")
    parser.add_argument("--verbose", action="store_true", help="Per-question detail")
    args = parser.parse_args()

    print("🔍 DocMind Retrieval Evaluation")
    print("=" * 50)

    with open(BENCHMARK, encoding="utf-8") as f:
        bench = json.load(f)

    corpus    = bench["corpus"]
    questions = bench["questions"]

    print(f"📚 Corpus: {len(corpus):,} chars  |  Questions: {len(questions)}")
    print("⏳ Loading embedding model (first run downloads ~90 MB)…\n")
    model = SentenceTransformer(EMBED_MODEL)

    print("── Fixed chunking ─────────────────────────────────")
    fixed = evaluate(corpus, questions, "fixed", model, verbose=args.verbose)

    print("\n── Sentence-aware chunking ────────────────────────")
    sentence = evaluate(corpus, questions, "sentence", model, verbose=args.verbose)

    print("\n" + "=" * 50)
    print("RESULTS\n")
    print(_markdown_table(fixed, sentence))

    delta3 = sentence["recall@3"] - fixed["recall@3"]
    delta5 = sentence["recall@5"] - fixed["recall@5"]
    sign3  = "+" if delta3 >= 0 else ""
    sign5  = "+" if delta5 >= 0 else ""
    print(f"\nSentence-aware vs Fixed  →  Recall@3: {sign3}{delta3:.0%}  |  Recall@5: {sign5}{delta5:.0%}")

    if args.save:
        out = Path(__file__).parent / "eval_results.md"
        _save_markdown(fixed, sentence, out)


if __name__ == "__main__":
    main()
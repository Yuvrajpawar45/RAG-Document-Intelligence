"""Compare fixed and sentence-aware chunk retrieval on the bundled benchmark."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.rag_engine import EMBED_MODEL, chunk_text

EVAL_DIR = ROOT / "eval"
STRATEGIES = ("fixed", "sentence")
TOP_K_VALUES = (3, 5)


def keyword_recall(retrieved_chunks: list[str], keywords: list[str]) -> float:
    retrieved_text = " ".join(retrieved_chunks).lower()
    matches = sum(keyword.lower() in retrieved_text for keyword in keywords)
    return matches / len(keywords)


def evaluate_strategy(model, document: str, benchmark: list[dict], strategy: str) -> dict:
    chunks = chunk_text(document, strategy)
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    question_embeddings = model.encode(
        [item["question"] for item in benchmark],
        normalize_embeddings=True,
    )

    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(np.asarray(chunk_embeddings, dtype="float32"))
    _, indices = index.search(
        np.asarray(question_embeddings, dtype="float32"),
        min(max(TOP_K_VALUES), len(chunks)),
    )

    result = {"strategy": strategy, "chunks": len(chunks)}
    for top_k in TOP_K_VALUES:
        recalls = [
            keyword_recall(
                [chunks[index] for index in row[:top_k]],
                item["ground_truth_keywords"],
            )
            for row, item in zip(indices, benchmark)
        ]
        result[f"recall_at_{top_k}"] = sum(recalls) / len(recalls)
    return result


def markdown_table(results: list[dict]) -> str:
    rows = [
        "# Chunking Benchmark Results",
        "",
        "Keyword recall averaged across 8 questions from `eval/benchmark.json`.",
        "",
        "| Strategy | Chunks | Recall@3 | Recall@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        rows.append(
            f"| {result['strategy'].title()} | {result['chunks']} | "
            f"{result['recall_at_3']:.3f} | {result['recall_at_5']:.3f} |"
        )
    rows.append("")
    return "\n".join(rows)


def main():
    document = (EVAL_DIR / "sample_doc.txt").read_text(encoding="utf-8")
    benchmark = json.loads((EVAL_DIR / "benchmark.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBED_MODEL)
    results = [
        evaluate_strategy(model, document, benchmark, strategy)
        for strategy in STRATEGIES
    ]

    print("\nChunking benchmark")
    print(f"{'Strategy':<12} {'Chunks':>8} {'Recall@3':>10} {'Recall@5':>10}")
    print("-" * 44)
    for result in results:
        print(
            f"{result['strategy']:<12} {result['chunks']:>8} "
            f"{result['recall_at_3']:>10.3f} {result['recall_at_5']:>10.3f}"
        )

    output = EVAL_DIR / "eval_results.md"
    output.write_text(markdown_table(results), encoding="utf-8")
    print(f"\nSaved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

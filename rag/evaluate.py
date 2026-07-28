"""
Evaluation harness (Final Project Brief, Section 2, Req 8).

Loads eval/test_queries.json — a set of test queries, each annotated with the
source files that SHOULD be retrieved (or marked "expect_refusal" for
off-domain queries) — and reports:

  hit@k             was at least one expected source in the top-k results?
  precision@k       what fraction of the top-k results were expected sources?
  MRR               mean reciprocal rank of the first expected source
  refusal accuracy  did off-domain queries correctly trigger the refusal gate?

Run:  python run_eval.py [--backend st|tfidf] [--top-k 3]
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

from .embed_store import VectorStore
from .generate import RELEVANCE_THRESHOLD
from .ingest import build_chunk_records, load_documents


@dataclass
class QueryResult:
    query: str
    expect_refusal: bool
    expected_sources: List[str]
    retrieved_sources: List[str] = field(default_factory=list)
    top_score: float = 0.0
    hit: bool = False
    precision: float = 0.0
    reciprocal_rank: float = 0.0
    refused: bool = False
    refusal_correct: bool = False


def run_evaluation(data_folder: str, eval_path: str, backend: str = "auto",
                   top_k: int = 3, chunk_size: int = 120) -> Dict:
    with open(eval_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    docs = load_documents(data_folder)
    chunks = build_chunk_records(docs, chunk_size=chunk_size)
    store = VectorStore(backend=backend)
    store.build(chunks)
    threshold = getattr(store.backend, "default_threshold", RELEVANCE_THRESHOLD)

    results: List[QueryResult] = []
    for item in test_queries:
        expected = item.get("expected_sources", [])
        r = QueryResult(query=item["query"],
                        expect_refusal=item.get("expect_refusal", False),
                        expected_sources=expected)

        retrieved = store.query(item["query"], top_k=top_k)
        r.retrieved_sources = [c.source_file for c, _ in retrieved]
        r.top_score = retrieved[0][1] if retrieved else 0.0
        r.refused = r.top_score < threshold

        if r.expect_refusal:
            r.refusal_correct = r.refused
        else:
            hits = [src in expected for src in r.retrieved_sources]
            r.hit = any(hits)
            r.precision = sum(hits) / max(len(hits), 1)
            for rank, is_hit in enumerate(hits, start=1):
                if is_hit:
                    r.reciprocal_rank = 1.0 / rank
                    break
            r.refusal_correct = not r.refused  # answerable query must not refuse
        results.append(r)

    answerable = [r for r in results if not r.expect_refusal]
    refusals = [r for r in results if r.expect_refusal]

    summary = {
        "backend": store.backend.name,
        "top_k": top_k,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "num_queries": len(results),
        f"hit@{top_k}": (sum(r.hit for r in answerable) / len(answerable)) if answerable else None,
        f"precision@{top_k}": (sum(r.precision for r in answerable) / len(answerable)) if answerable else None,
        "mrr": (sum(r.reciprocal_rank for r in answerable) / len(answerable)) if answerable else None,
        "refusal_accuracy": (sum(r.refusal_correct for r in refusals) / len(refusals)) if refusals else None,
    }
    return {"summary": summary, "results": results}


def print_report(report: Dict) -> None:
    s = report["summary"]
    print("=" * 64)
    print(f"RAG Evaluation — backend={s['backend']}  top_k={s['top_k']}")
    print(f"Corpus: {s['num_docs']} docs -> {s['num_chunks']} chunks")
    print("=" * 64)
    for r in report["results"]:
        tag = "REFUSAL-TEST" if r.expect_refusal else "QUERY"
        print(f"\n[{tag}] {r.query}")
        print(f"  top score: {r.top_score:.3f}   retrieved: {r.retrieved_sources}")
        if r.expect_refusal:
            print(f"  refused correctly: {r.refusal_correct}")
        else:
            print(f"  hit: {r.hit}   precision: {r.precision:.2f}   RR: {r.reciprocal_rank:.2f}")
    print("\n" + "-" * 64)
    for key in (f"hit@{s['top_k']}", f"precision@{s['top_k']}", "mrr", "refusal_accuracy"):
        value = s[key]
        print(f"{key:>18}: {value:.2%}" if value is not None else f"{key:>18}: n/a")
    print("-" * 64)

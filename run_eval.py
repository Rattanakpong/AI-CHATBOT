"""CLI entry point for the evaluation harness (Brief, Section 2, Req 8).

Usage:
    python run_eval.py                # auto backend (sentence-transformers if installed)
    python run_eval.py --backend tfidf --top-k 5
"""
import argparse

from rag.evaluate import print_report, run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "st", "tfidf", "openai"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--data", default="data/sme_docs")
    parser.add_argument("--eval", default="eval/test_queries.json")
    args = parser.parse_args()

    report = run_evaluation(args.data, args.eval, backend=args.backend, top_k=args.top_k)
    print_report(report)

"""
generate_predictions.py
-----------------------
Generates predictions.csv on the unlabeled test set.

Usage:
  python generate_predictions.py --test_path data/test.csv --output predictions.csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data_loader import build_or_load_indices
from core.retriever import SHLRetriever
from core.rag_graph import build_rag_graph, run_rag_pipeline


def load_test_queries(test_path: str):
    queries = []
    with open(test_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or row.get("Query") or "").strip()
            if q and q not in queries:
                queries.append(q)
    return queries


def generate_predictions(test_path: str, output_path: str):
    # Build pipeline
    faiss_store, bm25_index, docs = build_or_load_indices()
    retriever = SHLRetriever(faiss_store, bm25_index, docs)
    graph = build_rag_graph(retriever)

    queries = load_test_queries(test_path)
    print(f"Generating predictions for {len(queries)} test queries...")

    rows = []
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {query[:80]}...")
        try:
            result = run_rag_pipeline(query, graph)
            if result["abstained"]:
                print(f"  ABSTAINED: {result['abstention_reason'][:80]}")
                rows.append({"query": query, "Assessment_url": ""})
            else:
                for rec in result["recommendations"]:
                    rows.append({"query": query, "Assessment_url": rec["url"]})
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"query": query, "Assessment_url": ""})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/test.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    generate_predictions(args.test_path, args.output)

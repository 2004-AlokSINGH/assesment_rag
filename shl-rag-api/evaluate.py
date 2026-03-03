"""
evaluate.py
-----------
Evaluation utilities implementing Mean Recall@K as required by the assignment.

Usage:
  python evaluate.py --train_path data/train.csv --k 10
"""

from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data_loader import build_or_load_indices
from core.retriever import SHLRetriever
from core.rag_graph import build_rag_graph, run_rag_pipeline


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def recall_at_k(predicted_urls: List[str], relevant_urls: List[str], k: int) -> float:
    """Recall@K = |relevant ∩ predicted[:k]| / |relevant|"""
    if not relevant_urls:
        return 0.0
    predicted_top_k = set(predicted_urls[:k])
    relevant_set = set(relevant_urls)
    return len(predicted_top_k & relevant_set) / len(relevant_set)


def mean_recall_at_k(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    k: int = 10,
) -> float:
    scores = [
        recall_at_k(pred, gt, k)
        for pred, gt in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Load train CSV
# ---------------------------------------------------------------------------
def load_train_data(train_path: str) -> Dict[str, List[str]]:
    """
    Expected CSV format: query,relevant_url
    Returns {query: [url1, url2, ...]}
    """
    data: Dict[str, List[str]] = {}
    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("query", "").strip()
            url = row.get("relevant_url", "").strip() or row.get("Assessment_url", "").strip()
            if q and url:
                data.setdefault(q, []).append(url)
    return data


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(train_path: str, k: int = 10):
    print(f"\n{'='*60}")
    print(f"SHL RAG Evaluation | Recall@{k}")
    print(f"{'='*60}")

    # Build pipeline
    faiss_store, bm25_index, docs = build_or_load_indices()
    retriever = SHLRetriever(faiss_store, bm25_index, docs)
    graph = build_rag_graph(retriever)

    # Load ground truth
    train_data = load_train_data(train_path)
    queries = list(train_data.keys())
    print(f"Evaluating {len(queries)} queries...\n")

    all_predictions, all_ground_truths = [], []

    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query[:80]}...")
        try:
            result = run_rag_pipeline(query, graph)
            if result["abstained"]:
                print(f"  ABSTAINED (score {result['top_score']:.2f}): {result['abstention_reason'][:60]}")
                pred_urls = []
            else:
                pred_urls = [r["url"] for r in result["recommendations"]]
        except Exception as e:
            print(f"  ERROR: {e}")
            pred_urls = []

        gt_urls = train_data[query]
        r_at_k = recall_at_k(pred_urls, gt_urls, k)
        print(f"  Recall@{k}: {r_at_k:.3f} | Predicted {len(pred_urls)} | GT: {len(gt_urls)}")

        all_predictions.append(pred_urls)
        all_ground_truths.append(gt_urls)

    mean_r = mean_recall_at_k(all_predictions, all_ground_truths, k)
    print(f"\n{'='*60}")
    print(f"Mean Recall@{k}: {mean_r:.4f}")
    print(f"{'='*60}\n")
    return mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/train.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.train_path, args.k)

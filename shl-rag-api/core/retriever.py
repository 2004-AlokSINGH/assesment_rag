"""
retriever.py
------------
ADVANCED RAG TECHNIQUE 1: Hybrid Search (BM25 + FAISS + RRF)
ADVANCED RAG TECHNIQUE 2: Cross-Encoder Reranking
ADVANCED RAG TECHNIQUE 3: Corrective RAG (CRAG)
ADVANCED RAG TECHNIQUE 4: Lost-in-the-Middle Mitigation

See README_technical.md for full design rationale.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------------------------
# Cross-Encoder
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[Retriever] Loading CrossEncoder '{CROSS_ENCODER_MODEL}'...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _rrf_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Document],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """RRF score(d) = Σ 1 / (k + rank_i(d))"""
    scores:  Dict[str, float]    = {}
    doc_map: Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results):
        uid = doc.metadata.get("url", doc.page_content[:60])
        scores[uid]  = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc

    for rank, doc in enumerate(sparse_results):
        uid = doc.metadata.get("url", doc.page_content[:60])
        scores[uid]  = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[uid] = doc

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[uid], score) for uid, score in sorted_items]


# ---------------------------------------------------------------------------
# Lost-in-the-Middle mitigation
# ---------------------------------------------------------------------------
def _lost_in_middle_reorder(docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    LLMs tend to ignore documents placed in the middle of a long context window.
    This is the "Lost in the Middle" problem (Liu et al., 2023).

    Fix: place the MOST relevant docs at the START and END of the list,
    and less relevant ones in the middle. This way the LLM always sees the
    most important context at attention-strong positions.

    Pattern for N=10: positions → [1,3,5,7,9, 10,8,6,4,2] (alternating from ends inward)
    """
    if len(docs_with_scores) <= 3:
        return docs_with_scores

    # Split into two halves — first half goes to START, second goes to END (reversed)
    mid = len(docs_with_scores) // 2
    top_half    = docs_with_scores[:mid]
    bottom_half = docs_with_scores[mid:]

    reordered = []
    # Interleave: top at start, bottom reversed at end
    for i in range(max(len(top_half), len(bottom_half))):
        if i < len(top_half):
            reordered.append(top_half[i])
        if i < len(bottom_half):
            reordered.append(bottom_half[-(i + 1)])

    return reordered


# ---------------------------------------------------------------------------
# Relevance guard — hard filter on cross-encoder score
# ---------------------------------------------------------------------------
# Any document scoring below this absolute threshold is dropped entirely.
# ms-marco-MiniLM-L-6-v2 outputs logits (not probabilities); empirically:
#   score > 0    → clearly relevant
#   -3 to 0     → borderline
#   < -5         → clearly irrelevant (e.g. C++ when asking about writing)
RELEVANCE_FLOOR = -4.0


# ---------------------------------------------------------------------------
# SHLRetriever
# ---------------------------------------------------------------------------
class SHLRetriever:
    """
    Full pipeline:
      Hybrid Search (FAISS + BM25 + RRF)
        → Cross-Encoder Reranking
          → Relevance Floor Filter
            → CRAG Confidence Check
              → Lost-in-the-Middle Reorder
    """

    CONFIDENCE_THRESHOLD = -3.0   # tighter than before — ensures real relevance
    CANDIDATE_K          = 30     # candidates retrieved before reranking

    def __init__(
        self,
        faiss_store: FAISS,
        bm25_index:  BM25Okapi,
        all_docs:    List[Document],
    ):
        self.faiss_store   = faiss_store
        self.bm25_index    = bm25_index
        self.all_docs      = all_docs
        self.cross_encoder = _get_cross_encoder()

    # ------------------------------------------------------------------
    # Step 1: Hybrid retrieval
    # ------------------------------------------------------------------
    def _hybrid_retrieve(self, query: str, candidate_k: int) -> List[Tuple[Document, float]]:
        # Dense (BGE with instruction prefix for better asymmetric retrieval)
        from core.data_loader import BGE_QUERY_PREFIX
        prefixed_query = BGE_QUERY_PREFIX + query

        dense = self.faiss_store.similarity_search_with_score(prefixed_query, k=candidate_k)

        # Sparse (BM25) — use original query, no prefix needed
        tokens     = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokens)
        top_idx    = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_docs = [self.all_docs[i] for i in top_idx]

        fused = _rrf_fusion(dense, sparse_docs)
        return fused[:candidate_k]

    # ------------------------------------------------------------------
    # Step 2: Cross-encoder reranking
    # ------------------------------------------------------------------
    def _rerank(
        self, query: str, candidates: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        pairs    = [(query, doc.page_content) for doc, _ in candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip([doc for doc, _ in candidates], ce_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked

    # ------------------------------------------------------------------
    # Step 3: Relevance floor — hard filter
    # ------------------------------------------------------------------
    def _apply_relevance_floor(
        self, reranked: List[Tuple[Document, float]], min_k: int
    ) -> List[Tuple[Document, float]]:
        """
        Drop candidates whose cross-encoder score is below RELEVANCE_FLOOR.
        Always keep at least min_k results (even if below floor) to avoid
        empty responses on unusual queries.
        """
        above = [(doc, s) for doc, s in reranked if s >= RELEVANCE_FLOOR]
        if len(above) >= min_k:
            return above
        # Not enough above floor — return top min_k anyway with a warning
        print(f"[Retriever] Warning: only {len(above)} results above floor "
              f"({RELEVANCE_FLOOR}). Returning top {min_k} anyway.")
        return reranked[:min_k]

    # ------------------------------------------------------------------
    # Step 4: CRAG confidence check
    # ------------------------------------------------------------------
    def _crag_check(
        self, reranked: List[Tuple[Document, float]]
    ) -> Tuple[List[Tuple[Document, float]], bool]:
        if not reranked:
            return reranked, False
        is_confident = reranked[0][1] >= self.CONFIDENCE_THRESHOLD
        return reranked, is_confident

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10, min_k: int = 5) -> Dict[str, Any]:
        # 1. Hybrid search
        candidates = self._hybrid_retrieve(query, self.CANDIDATE_K)

        # 2. Cross-encoder rerank
        reranked = self._rerank(query, candidates)

        # 3. Hard relevance filter (removes truly irrelevant results)
        reranked = self._apply_relevance_floor(reranked, min_k)

        # 4. CRAG confidence check
        reranked, is_confident = self._crag_check(reranked)

        # 5. Slice to top_k
        top_results = reranked[:max(min_k, min(top_k, len(reranked)))]

        # 6. Lost-in-the-Middle reorder before passing to LLM
        top_results = _lost_in_middle_reorder(top_results)

        results = []
        for doc, score in top_results:
            m = doc.metadata
            results.append({
                "name":             m.get("name", ""),
                "url":              m.get("url", ""),
                "adaptive_support": m.get("adaptive_support", "No"),
                "description":      m.get("description", ""),
                "duration":         m.get("duration"),
                "remote_support":   m.get("remote_support", "No"),
                "test_type":        m.get("test_type_codes", []),
                "test_type_full":   m.get("test_type_full", []),
                "_rerank_score":    round(score, 4),
            })

        return {
            "results":       results,
            "is_confident":  is_confident,
            "top_score":     round(reranked[0][1], 4) if reranked else -999,
        }

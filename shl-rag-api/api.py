"""
api.py
------
FastAPI application exposing two endpoints as required by the assignment:

  GET  /health      → {"status": "healthy"}
  POST /recommend   → {"recommended_assessments": [...]} or abstention response
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.data_loader import build_or_load_indices
from core.retriever import SHLRetriever
from core.rag_graph import build_rag_graph, run_rag_pipeline


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
_retriever: Optional[SHLRetriever] = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _graph
    print("[API] Initializing RAG pipeline...")
    faiss_store, bm25_index, docs = build_or_load_indices()
    _retriever = SHLRetriever(faiss_store, bm25_index, docs)
    _graph     = build_rag_graph(_retriever)
    print("[API] RAG pipeline ready.")
    yield
    print("[API] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Industry-grade RAG with Hybrid Search, Reranking, CRAG, and Abstention",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    query: str


class AssessmentOut(BaseModel):
    url:              str
    name:             str
    adaptive_support: str
    description:      str
    duration:         Optional[int]
    remote_support:   str
    test_type:        List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentOut]
    abstained:               bool  = False
    abstention_reason:       str   = ""
    retrieval_top_score:     float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    if _graph is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = run_rag_pipeline(query, _graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- Abstention case ---
    if result["abstained"]:
        return RecommendResponse(
            recommended_assessments=[],
            abstained=True,
            abstention_reason=result["abstention_reason"],
            retrieval_top_score=result["top_score"],
        )

    recommendations = result["recommendations"][:10]
    if not recommendations:
        raise HTTPException(status_code=404, detail="No relevant assessments found")

    out = [
        AssessmentOut(
            url=r.get("url", ""),
            name=r.get("name", ""),
            adaptive_support=r.get("adaptive_support", "No"),
            description=r.get("description", ""),
            duration=r.get("duration"),
            remote_support=r.get("remote_support", "No"),
            test_type=r.get("test_type", []),
        )
        for r in recommendations
    ]

    return RecommendResponse(
        recommended_assessments=out,
        abstained=False,
        abstention_reason="",
        retrieval_top_score=result["top_score"],
    )

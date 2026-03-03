"""
rag_graph.py
------------
LangGraph orchestration for the SHL recommendation pipeline.

Graph nodes:
  1. query_analyzer      – LLM extracts skills, job levels, test types from query
  2. retriever_node      – Calls SHLRetriever (hybrid + rerank + CRAG)
  3. crag_router         – Conditional: confident → formatter | low → refiner | dead → abstain
  4. query_refiner_node  – LLM reformulates query if CRAG flags low confidence (1 retry)
  5. abstention_node     – Returns explicit "no suitable assessments found" when even after
                           retry the top_score is still below HARD_ABSTENTION_THRESHOLD
  6. formatter_node      – LLM selects 5-10 recommendations with explicit abstention instruction

Hallucination defences implemented here:
  ✅ CRAG confidence threshold    (Post-Retrieval)
  ✅ Explicit "I don't know"      (LangGraph abstention_node + prompt instruction)
  ✅ Retrieval score display      (scores shown in candidate list passed to LLM)
  ✅ Abstention instruction       (formatter prompt rule 7)
"""

from __future__ import annotations
import json
import os
import re
from functools import partial
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.retriever import SHLRetriever

load_dotenv()

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# After ONE retry, if the best reranker score is still below this value
# the pipeline routes to the abstention node instead of the formatter.
# The formatter is skipped entirely — no LLM hallucination risk on bad retrieval.
HARD_ABSTENTION_THRESHOLD = -6.0

# Response object returned by abstention node so callers get a consistent shape
ABSTENTION_RESPONSE = {
    "abstained": True,
    "message": (
        "No sufficiently relevant SHL assessments were found for this query. "
        "The query may be too vague, refer to a role not well covered by the catalog, "
        "or use terminology that does not match any assessment description. "
        "Please try rephrasing with specific skills, job level, or assessment type."
    ),
    "top_score": None,
}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    original_query:       str
    refined_query:        str
    extracted_intent:     str                   # JSON string of skills / types
    raw_results:          List[Dict[str, Any]]
    is_confident:         bool
    top_score:            float
    final_recommendations: List[Dict[str, Any]]
    abstained:            bool                  # True when pipeline deliberately abstains
    abstention_reason:    str                   # Human-readable reason for abstention
    retry_count:          int


# ---------------------------------------------------------------------------
# Node 1 — Query Analyzer
# ---------------------------------------------------------------------------
def query_analyzer_node(state: RAGState) -> RAGState:
    """
    Extract structured intent from the natural-language query so that:
      - downstream retrieval can use a cleaner search string
      - the formatter knows which test types are expected
      - the abstention node can give a meaningful reason if no results found
    """
    llm   = _get_llm()
    query = state["original_query"]

    system = (
        "You are an HR assessment expert. Given a hiring query or job description, "
        "extract the following as a JSON object:\n"
        "{\n"
        '  "key_skills": [list of technical and soft skills mentioned],\n'
        '  "job_levels": [e.g. "Entry-Level", "Mid-Professional", "Manager"],\n'
        '  "test_types_needed": [one or more of: "A" (Ability & Aptitude), '
        '"B" (Biodata & SJT), "C" (Competencies), "D" (Development & 360), '
        '"E" (Assessment Exercises), "K" (Knowledge & Skills), '
        '"P" (Personality & Behavior), "S" (Simulations)],\n'
        '  "search_query": "a concise retrieval-optimized reformulation of the query"\n'
        "}\n"
        "Return ONLY the JSON. No markdown, no extra text."
    )

    response    = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
    intent_text = response.content.strip()

    return {
        **state,
        "extracted_intent": intent_text,
        "refined_query":    query,          # may be overwritten by query_refiner
    }


# ---------------------------------------------------------------------------
# Node 2 — Retriever
# ---------------------------------------------------------------------------
def retriever_node(state: RAGState, retriever: SHLRetriever) -> RAGState:
    """Run hybrid search + reranking + CRAG on the current refined_query."""
    query  = state.get("refined_query") or state["original_query"]
    output = retriever.retrieve(query, top_k=10, min_k=5)

    return {
        **state,
        "raw_results":  output["results"],
        "is_confident": output["is_confident"],
        "top_score":    output["top_score"],
    }


# ---------------------------------------------------------------------------
# Node 3 — Query Refiner  (CRAG retry)
# ---------------------------------------------------------------------------
def query_refiner_node(state: RAGState) -> RAGState:
    """
    CRAG fallback: LLM rewrites the query to improve retrieval quality.
    Only executed when CRAG flags low confidence on the first pass.
    """
    llm = _get_llm()

    system = (
        "You are a search query optimizer for an HR assessment catalog. "
        "The previous retrieval had low confidence — the results were not relevant enough. "
        "Rewrite the query to be more specific and retrieval-friendly. "
        "Focus on concrete skill names, assessment types, and job context. "
        "Return ONLY the improved query string, no explanation."
    )
    human = (
        f"Original query: {state['original_query']}\n"
        f"Extracted intent: {state['extracted_intent']}\n"
        f"Previous retrieval top score: {state['top_score']}\n"
        "Write an improved retrieval query:"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return {
        **state,
        "refined_query": response.content.strip(),
        "retry_count":   state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 4 — Abstention  (NEW — explicit "I don't know")
# ---------------------------------------------------------------------------
def abstention_node(state: RAGState) -> RAGState:
    """
    Explicit abstention node — reached when:
      • CRAG already triggered one query refinement AND
      • The post-retry top_score is still below HARD_ABSTENTION_THRESHOLD

    Instead of forcing the LLM to hallucinate recommendations from poor context,
    we short-circuit here and return a structured "no suitable assessments" response.

    This implements the "Explicit I don't know fallback" from the hallucination
    prevention strategy. The pipeline never reaches formatter_node in this case,
    so there is zero chance of generation hallucination on a bad retrieval.

    The reason is logged so the caller (API / Streamlit) can surface a helpful
    message to the recruiter rather than a silent empty list.
    """
    # Build a human-readable reason using the extracted intent
    intent = state.get("extracted_intent", "")
    try:
        intent_dict = json.loads(intent)
        skills = ", ".join(intent_dict.get("key_skills", [])) or "unspecified skills"
        levels = ", ".join(intent_dict.get("job_levels", [])) or "unspecified level"
    except Exception:
        skills = "unspecified skills"
        levels = "unspecified level"

    reason = (
        f"After retrieval and one query refinement attempt, no SHL assessments scored "
        f"above the minimum confidence threshold (top score: {state['top_score']:.2f}, "
        f"required: {HARD_ABSTENTION_THRESHOLD}). "
        f"The query appears to be asking about '{skills}' at '{levels}' level, "
        f"which may not be well-covered by the current catalog. "
        f"Suggestions: use more specific skill terms, specify test type (e.g. 'personality test', "
        f"'cognitive ability test'), or check whether the role has a matching assessment."
    )

    print(f"[RAGGraph] ABSTAINING — {reason}")

    return {
        **state,
        "abstained":           True,
        "abstention_reason":   reason,
        "final_recommendations": [],    # empty — no hallucinated results
    }


# ---------------------------------------------------------------------------
# Node 5 — Formatter  (with abstention instruction added)
# ---------------------------------------------------------------------------
def formatter_node(state: RAGState) -> RAGState:
    """
    Final node: LLM selects the best 5-10 assessments from raw_results.

    Hallucination controls active here:
      ✅ Retrieval scores shown to LLM  → LLM can reason about confidence
      ✅ Abstention instruction (Rule 7) → LLM can say "none fit" rather than hallucinate
      ✅ Lost-in-Middle reorder          → done upstream in retriever
      ✅ URL validation post-response    → invented URLs are hard-dropped
      ✅ Fallback to raw_results         → if LLM output is malformed
    """
    llm = _get_llm()

    # ------------------------------------------------------------------
    # Build candidate text — rerank score is explicitly shown so the LLM
    # can reason: "score -7 = probably irrelevant, I should skip this one"
    # ------------------------------------------------------------------
    candidates_text = "\n".join(
        f"{i+1}. [Relevance Score: {r.get('_rerank_score', '?'):>7}] "
        f"Name: {r['name']} | "
        f"Category: {', '.join(r.get('test_type_full', r.get('test_type', [])))} | "
        f"Description: {r['description'][:250]} | "
        f"Remote: {r['remote_support']} | Adaptive: {r['adaptive_support']} | "
        f"Duration: {r['duration']} min | URL: {r['url']}"
        for i, r in enumerate(state["raw_results"])
    )

    # Context about overall retrieval confidence — gives LLM a signal on
    # whether to be selective or lenient
    confidence_context = (
        f"Overall retrieval confidence: {'HIGH' if state['is_confident'] else 'LOW'} "
        f"(best score: {state['top_score']})"
    )

    system = (
        "You are an SHL assessment recommendation expert. "
        "Given a hiring query and candidate assessments pre-ranked by relevance score, "
        "select the MOST RELEVANT ones to recommend.\n\n"
        "STRICT RULES:\n"
        "1. ONLY recommend assessments that are clearly relevant to the query. "
        "   If the query is about writing or communication, do NOT include programming/coding tests.\n"
        "2. Balance test categories when the query needs multiple skill types "
        "   (technical skills → Knowledge & Skills; soft skills → Personality & Behavior).\n"
        "3. Return EXACTLY a JSON object (no markdown fences, no extra text):\n"
        "   If you found good matches:\n"
        '   {"found": true, "recommendations": [{"name": str, "url": str, '
        '"adaptive_support": str, "description": str, "duration": int|null, '
        '"remote_support": str, "test_type": [str], "test_type_full": [str]}]}\n'
        "   If NO candidates are genuinely relevant to the query:\n"
        '   {"found": false, "reason": "brief explanation why nothing fits"}\n'
        "4. Select between 5 and 10 assessments when found=true.\n"
        "5. NEVER invent assessments. Only use the provided candidate list.\n"
        "6. Higher Relevance Score = more relevant. Scores below -5 are likely irrelevant.\n"
        "7. ABSTENTION RULE — if the query is clearly outside the scope of all provided "
        "   candidates (e.g. asking for an assessment type that none of them cover), "
        "   return found=false with a clear reason rather than forcing a poor recommendation."
    )

    human = (
        f"Hiring Query: {state['original_query']}\n"
        f"Extracted Intent: {state['extracted_intent']}\n"
        f"{confidence_context}\n\n"
        f"Candidate Assessments (most relevant at start and end):\n{candidates_text}\n\n"
        "Return the JSON object:"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    raw_text  = response.content.strip()

    # ------------------------------------------------------------------
    # Parse + validate LLM response
    # ------------------------------------------------------------------
    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        parsed = json.loads(clean)

        # --- Case A: LLM chose to abstain ---
        if not parsed.get("found", True):
            reason = parsed.get("reason", "LLM determined no candidates were relevant.")
            print(f"[RAGGraph] LLM abstained: {reason}")
            return {
                **state,
                "abstained":            True,
                "abstention_reason":    f"LLM abstention: {reason}",
                "final_recommendations": [],
            }

        # --- Case B: LLM returned recommendations ---
        selected = parsed.get("recommendations", [])
        if not isinstance(selected, list):
            raise ValueError("recommendations is not a list")

        # Hard validation — drop any URL not in the retrieved set (prevents hallucination)
        valid_urls = {r["url"] for r in state["raw_results"]}
        selected   = [s for s in selected if s.get("url") in valid_urls]

        # Enforce minimum — if LLM filtered too aggressively, pad with top raw results
        if len(selected) < 5:
            existing_urls = {s["url"] for s in selected}
            for r in state["raw_results"]:
                if r["url"] not in existing_urls:
                    selected.append(r)
                if len(selected) >= 5:
                    break

    except Exception as e:
        print(f"[RAGGraph] formatter parse error: {e} — falling back to raw_results")
        selected = state["raw_results"][:10]

    return {
        **state,
        "abstained":             False,
        "abstention_reason":     "",
        "final_recommendations": selected,
    }


# ---------------------------------------------------------------------------
# Conditional edge — CRAG router  (now has 3 paths)
# ---------------------------------------------------------------------------
def crag_router(state: RAGState) -> str:
    """
    Three-way routing:

    Path 1 → formatter_node
      Condition: retrieval was confident (top_score ≥ CONFIDENCE_THRESHOLD)
      OR we already did one retry (don't loop forever)
      AND the score is at least above HARD_ABSTENTION_THRESHOLD

    Path 2 → query_refiner_node
      Condition: low confidence on first pass (retry_count == 0)
      Give the pipeline one chance to improve retrieval before giving up.

    Path 3 → abstention_node
      Condition: already retried AND score still below HARD_ABSTENTION_THRESHOLD
      No point asking the LLM to recommend from irrelevant candidates.
      Return an explicit structured "no match" response.
    """
    already_retried = state.get("retry_count", 0) >= 1

    # Hard floor — if even after retry the score is terrible, abstain
    if already_retried and state["top_score"] < HARD_ABSTENTION_THRESHOLD:
        print(
            f"[CRAGRouter] top_score={state['top_score']} < hard floor {HARD_ABSTENTION_THRESHOLD} "
            f"after retry — routing to abstention_node"
        )
        return "abstention_node"

    # Confident retrieval or already retried with acceptable score → format
    if state["is_confident"] or already_retried:
        return "formatter_node"

    # First pass, low confidence → refine and retry
    print(
        f"[CRAGRouter] top_score={state['top_score']} below confidence threshold "
        f"on first pass — routing to query_refiner_node"
    )
    return "query_refiner_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_rag_graph(retriever: SHLRetriever) -> Any:
    """Build and compile the LangGraph RAG pipeline."""
    bound_retriever = partial(retriever_node, retriever=retriever)

    graph = StateGraph(RAGState)

    graph.add_node("query_analyzer",     query_analyzer_node)
    graph.add_node("retriever_node",     bound_retriever)
    graph.add_node("query_refiner_node", query_refiner_node)
    graph.add_node("abstention_node",    abstention_node)       # NEW
    graph.add_node("formatter_node",     formatter_node)

    graph.add_edge(START, "query_analyzer")
    graph.add_edge("query_analyzer", "retriever_node")
    graph.add_conditional_edges(
        "retriever_node",
        crag_router,
        {
            "formatter_node":     "formatter_node",
            "query_refiner_node": "query_refiner_node",
            "abstention_node":    "abstention_node",    # NEW path
        },
    )
    graph.add_edge("query_refiner_node", "retriever_node")
    graph.add_edge("abstention_node",    END)                   # NEW — short-circuit to END
    graph.add_edge("formatter_node",     END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def run_rag_pipeline(query: str, compiled_graph) -> Dict[str, Any]:
    """
    Execute the RAG graph and return a result dict with shape:
      {
        "recommendations": List[dict],   # empty if abstained
        "abstained": bool,
        "abstention_reason": str,        # non-empty only when abstained
        "top_score": float,
      }
    """
    initial_state: RAGState = {
        "original_query":        query,
        "refined_query":         query,
        "extracted_intent":      "",
        "raw_results":           [],
        "is_confident":          False,
        "top_score":             -999.0,
        "final_recommendations": [],
        "abstained":             False,
        "abstention_reason":     "",
        "retry_count":           0,
    }

    result = compiled_graph.invoke(initial_state)

    return {
        "recommendations":  result.get("final_recommendations", []),
        "abstained":        result.get("abstained", False),
        "abstention_reason": result.get("abstention_reason", ""),
        "top_score":        result.get("top_score", -999.0),
    }

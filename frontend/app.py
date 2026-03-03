"""
app.py — SHL Assessment Recommender
Clean, professional UI using only native Streamlit components.
"""

import streamlit as st
import requests
import json
import time
from collections import Counter

API_URL = "https://alokxsingh-shl-rag-api.hf.space"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only safe overrides that Streamlit respects reliably
st.markdown("""
<style>
.block-container { padding-top: 2rem !important; max-width: 1100px !important; }
div[data-testid="metric-container"] { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & SJT",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

EXAMPLES = [
    "I am hiring Java developers who collaborate with business teams.",
    "Mid-level professionals proficient in Python, SQL and JavaScript.",
    "Data analyst role — assess analytical thinking and cognitive ability.",
    "Customer service manager with strong interpersonal and leadership skills.",
    "Assess writing proficiency and communication skills for a content writer.",
    "Entry-level sales executive with high resilience and persuasion skills.",
]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SHL RAG System")
    st.caption("v2.0 · LangGraph + LLaMA-3.3-70b")
    st.divider()

    st.markdown("**Pipeline**")
    for item in [
        "Hybrid Search (FAISS + BM25)",
        "Reciprocal Rank Fusion",
        "Cross-Encoder Reranking",
        "Corrective RAG (CRAG)",
        "Lost-in-Middle Reorder",
        "Abstention Guard",
    ]:
        st.caption(f"• {item}")

    st.divider()
    if st.button("Check API Health", use_container_width=True):
        with st.spinner("Checking..."):
            try:
                r = requests.get(f"{API_URL}/health", timeout=15)
                if r.status_code == 200:
                    st.success("API is healthy")
                else:
                    st.error(f"Status {r.status_code}")
            except Exception as e:
                st.error(f"Unreachable: {e}")

    st.divider()
    st.markdown("**Assessment Types**")
    for code, name in TYPE_MAP.items():
        st.caption(f"`{code}` — {name}")

# ── Main ─────────────────────────────────────────────────────
st.markdown("# SHL Assessment Recommender")
st.caption("Enter a job description or hiring query to get the most relevant SHL assessments.")
st.divider()

col_query, col_pick = st.columns([3, 1])

with col_pick:
    st.markdown("**Quick examples**")
    selected = st.selectbox(
        "Pick an example",
        ["— select —"] + EXAMPLES,
        label_visibility="collapsed",
    )

with col_query:
    st.markdown("**Query / Job Description**")
    query = st.text_area(
        "Query",
        value="" if selected == "— select —" else selected,
        height=130,
        placeholder="e.g. Hiring a Java developer who collaborates closely with business stakeholders...",
        label_visibility="collapsed",
    )

col_btn, col_hint = st.columns([2, 4])
with col_btn:
    submit = st.button("Get Recommendations", type="primary", use_container_width=True)
with col_hint:
    st.caption("First request may take ~30s if the API is waking up.")

st.divider()

# ── Results ──────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a query before submitting.")
        st.stop()

    with st.spinner("Running pipeline — Hybrid Search → Reranking → CRAG → LLM..."):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_URL}/recommend",
                json={"query": query.strip()},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = round(time.time() - t0, 1)
            assessments = data.get("recommended_assessments", [])
            error_msg = None
        except requests.HTTPError as e:
            data, assessments = {}, []
            elapsed = round(time.time() - t0, 1)
            error_msg = f"API error {e.response.status_code}: {e.response.text[:300]}"
        except Exception as e:
            data, assessments = {}, []
            elapsed = round(time.time() - t0, 1)
            error_msg = str(e)

    if error_msg:
        st.error(f"Request failed: {error_msg}")
        st.stop()

    if data.get("abstained"):
        st.warning("No suitable assessments found for this query.")
        st.info(data.get("abstention_reason", ""))
        st.caption(f"Best retrieval score: {data.get('retrieval_top_score', 'N/A')} · {elapsed}s")
        st.stop()

    if not assessments:
        st.warning("No assessments returned.")
        st.stop()

    # ── Metrics row ──
    all_types = []
    for a in assessments:
        all_types.extend(a.get("test_type", []))
    type_counts = Counter(all_types)
    remote_count   = sum(1 for a in assessments if a.get("remote_support", "No") == "Yes")
    adaptive_count = sum(1 for a in assessments if a.get("adaptive_support", "No") == "Yes")
    durations      = [a["duration"] for a in assessments if a.get("duration")]
    avg_dur        = round(sum(durations) / len(durations)) if durations else None
    top_score      = data.get("retrieval_top_score", "—")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Assessments", len(assessments), help="Total returned")
    m2.metric("Remote Ready", remote_count,    help="Support remote testing")
    m3.metric("Adaptive",     adaptive_count,  help="Use adaptive/IRT scoring")
    m4.metric("Avg Duration", f"{avg_dur} min" if avg_dur else "—", help="Average test length")
    m5.metric("Top Score",    top_score,        help=f"Pipeline ran in {elapsed}s")

    st.divider()

    # ── Type breakdown ──
    if type_counts:
        st.markdown("**Test Type Breakdown**")
        tcols = st.columns(len(type_counts))
        for col, (code, cnt) in zip(tcols, sorted(type_counts.items(), key=lambda x: -x[1])):
            col.metric(label=f"{code} — {TYPE_MAP.get(code, code)}", value=cnt)
        st.divider()

    # ── Assessment cards ──
    st.markdown(f"**Recommended Assessments** — {len(assessments)} results")
    st.write("")

    for i, a in enumerate(assessments):
        name     = a.get("name", "Unknown")
        desc     = a.get("description", "No description available.")
        url      = a.get("url", "")
        remote   = a.get("remote_support", "No")
        adaptive = a.get("adaptive_support", "No")
        duration = a.get("duration")
        types    = a.get("test_type", [])
        score_v  = a.get("_rerank_score", "")

        # Build a clean label for the expander
        type_labels = ", ".join(TYPE_MAP.get(t, t) for t in types)
        expander_label = f"#{i+1:02d}  {name}  ·  {type_labels}"

        with st.expander(expander_label, expanded=(i < 3)):
            # Meta row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Remote Testing", remote)
            c2.metric("Adaptive / IRT", adaptive)
            c3.metric("Duration", f"{duration} min" if duration else "—")
            c4.metric("Relevance Score", score_v if score_v else "—")

            st.write("")
            st.markdown(f"**Description**")
            st.write(desc)

            if url:
                st.markdown(f"[View on SHL Catalog]({url})")

    st.divider()

    # ── Raw JSON ──
    with st.expander("Raw JSON Response", expanded=False):
        st.json(data)

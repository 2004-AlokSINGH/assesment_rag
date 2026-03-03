"""
app.py
------
Streamlit frontend for the SHL Assessment Recommendation System.
Run with:
  streamlit run app.py
"""

import streamlit as st
import requests
import json

# Read from Streamlit secrets when deployed, fall back to localhost for local dev
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 SHL Assessment Recommendation System")
st.markdown(
    "Enter a **job description** or **natural language query** to get the most relevant SHL assessments."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_base = st.text_input("API Base URL", value=API_URL)
    st.markdown("---")
    st.markdown("**Advanced RAG Features:**")
    st.markdown("✅ Hybrid Search (BM25 + FAISS)")
    st.markdown("✅ Cross-Encoder Reranking")
    st.markdown("✅ Corrective RAG (CRAG)")
    st.markdown("✅ LangGraph Orchestration")
    st.markdown("---")

    # Health check
    if st.button("🩺 Check API Health"):
        try:
            r = requests.get(f"{api_base}/health", timeout=5)
            if r.status_code == 200:
                st.success("API is healthy ✅")
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
example_queries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "Need to assess analytical thinking and cognitive ability for a data analyst role.",
    "Hiring a customer service manager who needs strong interpersonal and leadership skills.",
]

col1, col2 = st.columns([3, 1])
with col2:
    selected_example = st.selectbox("Or pick an example:", [""] + example_queries)

with col1:
    query = st.text_area(
        "Your query / job description:",
        value=selected_example,
        height=150,
        placeholder="e.g. Need a Java developer who is good in collaborating with external teams...",
    )

submit = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submit and query.strip():
    with st.spinner("Running RAG pipeline (Hybrid Search → Reranking → CRAG → LLM)..."):
        try:
            response = requests.post(
                f"{api_base}/recommend",
                json={"query": query},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            assessments = data.get("recommended_assessments", [])
        except requests.HTTPError as e:
            st.error(f"API Error {e.response.status_code}: {e.response.text}")
            assessments = []
        except Exception as e:
            st.error(f"Request failed: {e}")
            assessments = []
            data = {}

    # --- Abstention case ---
    if data.get("abstained"):
        st.warning("⚠️ No sufficiently relevant assessments found")
        st.info(data.get("abstention_reason", ""))
        score = data.get("retrieval_top_score", "N/A")
        st.caption(f"Best retrieval score: {score} (threshold not met)")
        st.stop()

    if assessments:
        st.success(f"✅ Found {len(assessments)} relevant assessments  |  Top retrieval score: {data.get('retrieval_top_score', 'N/A')}")

        # Test type breakdown
        all_types = []
        for a in assessments:
            all_types.extend(a.get("test_type", []))
        if all_types:
            from collections import Counter
            type_counts = Counter(all_types)
            type_map = {
                "A": "Ability & Aptitude",
                "B": "Biodata & SJT",
                "C": "Competencies",
                "D": "Development & 360",
                "E": "Assessment Exercises",
                "K": "Knowledge & Skills",
                "P": "Personality & Behavior",
                "S": "Simulations",
            }
            st.markdown("**Test Type Balance:**")
            cols = st.columns(len(type_counts))
            for col, (t, cnt) in zip(cols, type_counts.items()):
                col.metric(label=type_map.get(t, t), value=cnt)

        st.markdown("---")

        for i, assessment in enumerate(assessments):
            with st.expander(
                f"**{i+1}. {assessment.get('name', 'N/A')}**  "
                f"{'  '.join(['`'+t+'`' for t in assessment.get('test_type', [])])}",
                expanded=(i < 3),
            ):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Remote Support", assessment.get("remote_support", "N/A"))
                col_b.metric("Adaptive/IRT", assessment.get("adaptive_support", "N/A"))
                duration = assessment.get("duration")
                col_c.metric("Duration", f"{duration} min" if duration else "N/A")

                st.markdown(f"**Description:** {assessment.get('description', 'N/A')}")
                url = assessment.get("url", "")
                if url:
                    st.markdown(f"🔗 [View on SHL Catalog]({url})")

        # Raw JSON toggle
        with st.expander("📄 Raw JSON Response"):
            st.json(data)

elif submit and not query.strip():
    st.warning("Please enter a query before submitting.")

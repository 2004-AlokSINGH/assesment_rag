FROM python:3.11-slim

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────────
# Copy requirements first so Docker layer cache works:
# if only code changes (not requirements), this layer is reused.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY . .

# ── PYTHONPATH ───────────────────────────────────────────────────────────────
# Ensure `from core.xxx import` resolves correctly when uvicorn runs api.py.
# Without this, Python cannot find the `core` package inside /app.
ENV PYTHONPATH=/app

# ── Pre-download models at BUILD time ────────────────────────────────────────
# Baking models into the image means:
#   • Zero model-download delay on container start
#   • No network dependency at runtime
#   • Reproducible: model version is pinned to what was downloaded during build
# HuggingFace caches to ~/.cache/huggingface by default inside the image.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading BGE embedding model (BAAI/bge-base-en-v1.5)...'); \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
print('Downloading CrossEncoder reranker...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('All models downloaded and cached.') \
"

# ── Port ─────────────────────────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860.
EXPOSE 7860

# ── Runtime ──────────────────────────────────────────────────────────────────
# GROQ_API_KEY is injected at runtime via HF Space Secrets (never hardcode it).
# On first start: FAISS + BM25 indices build from catalog (~2 min).
# On subsequent starts: indices load from disk (fast).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

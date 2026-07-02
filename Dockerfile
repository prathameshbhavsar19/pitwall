# ── Base image ────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies for sentence-transformers + PyPDF2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model ──────────────────────────────────────
# Bakes bge-large into the image so startup is fast (no download at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# ── Application code ──────────────────────────────────────────────────
COPY api/ ./api/
COPY data/chunks.json ./data/chunks.json

# ── Runtime ───────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

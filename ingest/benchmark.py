# benchmark.py — Day 2
# Compare 3 chunking strategies x 2 models = 6 combinations
# Tells you which combo gives the best retrieval scores

import PyPDF2
import json
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── YOUR PDF PATHS ────────────────────────────────────────
PDF_PATHS = [
    "data/fia_technical_regulations_2024.pdf",
    "data/fia_sporting_regulations_2024.pdf",
]

# ── MODELS TO BENCHMARK ───────────────────────────────────
MODELS = [
    "all-MiniLM-L6-v2",       # fast, 384-dim  (you used this Day 1)
    "BAAI/bge-large-en-v1.5", # slow, 1024-dim (production quality)
]

# ── TEST QUERIES (same 10 from Day 1) ────────────────────
QUERIES = [
    "What is the penalty for unsafe release in the pit lane?",
    "What is the maximum allowed front wing width?",
    "How many power unit components are allowed per season?",
    "What are the rules around tyre compounds at a race weekend?",
    "What is the minimum car weight including the driver?",
    "What happens if a driver misses the weigh bridge?",
    "What are the DRS activation zone rules?",
    "What is the fuel flow rate limit during a race?",
    "What defines a safety car period?",
    "What are the pit lane speed limit rules?",
]


# ─────────────────────────────────────────────────────────
# STEP 1 — Extract raw text from PDFs (same as Day 1)
# ─────────────────────────────────────────────────────────

def extract_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


# ─────────────────────────────────────────────────────────
# STEP 2 — 3 Chunking Strategies
# ─────────────────────────────────────────────────────────

def chunk_fixed(text, size=500, overlap=50):
    """
    Strategy 1: Fixed character size (Day 1 approach)
    Slides a window of 500 chars, moves forward 450 each time.
    Problem: cuts mid-sentence constantly.
    """
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks


def chunk_sentence(text, sentences_per_chunk=5):
    """
    Strategy 2: Sentence boundary chunking
    Uses nltk to detect sentence endings properly.
    Groups every N sentences into one chunk.
    Never cuts mid-sentence — cleaner meaning per chunk.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_paragraph(text, min_length=100):
    """
    Strategy 3: Paragraph chunking
    Splits on double newlines (natural paragraph boundaries in PDFs).
    Respects document structure — each chunk = one regulation article.
    Skips paragraphs shorter than min_length (headers, page numbers).
    """
    paragraphs = text.split("\n\n")
    chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    return chunks


# ─────────────────────────────────────────────────────────
# STEP 3 — Embed + Retrieve (same logic as Day 1)
# ─────────────────────────────────────────────────────────

def embed(chunks, model):
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def avg_top1_score(queries, chunks, embeddings, model):
    """
    For each query, get the top-1 cosine similarity score.
    Average across all queries = one number representing retrieval quality.
    Higher = better.
    """
    scores = []
    for query in queries:
        q_vec = model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, embeddings)[0]
        scores.append(float(np.max(sims)))
    return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────────────────
# STEP 4 — Run all 6 combinations and print results
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📄 Extracting text from PDFs...")
    text = extract_text(PDF_PATHS)
    print(f"✅ {len(text):,} characters extracted\n")

    strategies = {
        "fixed":     chunk_fixed(text),
        "sentence":  chunk_sentence(text),
        "paragraph": chunk_paragraph(text),
    }

    for name, chunks in strategies.items():
        print(f"  {name:12} → {len(chunks)} chunks")

    print("\n" + "="*62)
    print(f"{'Strategy':<12} {'Model':<26} {'Chunks':>7} {'Avg Top-1':>10} {'Time':>6}")
    print("="*62)

    results = []

    for model_name in MODELS:
        print(f"\n⚙️  Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        for strat_name, chunks in strategies.items():
            start = time.time()
            embeddings = embed(chunks, model)
            score = avg_top1_score(QUERIES, chunks, embeddings, model)
            elapsed = round(time.time() - start, 1)

            short_model = model_name.split("/")[-1]
            print(f"  {strat_name:<12} {short_model:<26} {len(chunks):>7} {score:>10} {elapsed:>5}s")

            results.append({
                "strategy": strat_name,
                "model": short_model,
                "chunks": len(chunks),
                "avg_top1": score,
                "time_s": elapsed
            })

    print("\n" + "="*62)
    best = max(results, key=lambda x: x["avg_top1"])
    print(f"🏆 Best combo: {best['strategy']} + {best['model']} → score {best['avg_top1']}")
    print("="*62)

    with open("data/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved → data/benchmark_results.json")
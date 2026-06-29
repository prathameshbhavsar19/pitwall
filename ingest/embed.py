# embed.py — Updated Day 2
# Model: bge-large-en-v1.5 (winner from benchmark, score 0.7242)

from sentence_transformers import SentenceTransformer
import numpy as np
import json

MODEL_NAME = "BAAI/bge-large-en-v1.5"   # upgraded from MiniLM


def load_chunks(path="data/chunks.json") -> list[dict]:
    with open(path) as f:
        chunks = json.load(f)
    print(f"📦 {len(chunks)} chunks loaded")
    return chunks


def embed_and_save(chunks: list[dict]):
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]

    print(f"⚙️  Embedding {len(texts)} chunks with {MODEL_NAME}...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # shape → (num_chunks, 1024) — note 1024 not 384, bge-large is wider

    np.save("data/embeddings.npy", embeddings)
    with open("data/chunks_text.json", "w") as f:
        json.dump(texts, f)

    print(f"✅ Saved embeddings {embeddings.shape} → data/embeddings.npy")


if __name__ == "__main__":
    chunks = load_chunks()
    embed_and_save(chunks)
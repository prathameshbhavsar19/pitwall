# qdrant_ingest.py — Day 3
# Takes your existing chunks + embeddings and loads them into Qdrant
# Replaces the flat .npy file with a proper indexed vector database

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import json

# ── CONFIG ────────────────────────────────────────────────
COLLECTION_NAME = "pitwall"
EMBEDDING_DIM   = 1024          # bge-large outputs 1024-dim vectors
QDRANT_URL      = "http://localhost:6333"

EMBEDDINGS_PATH    = "data/embeddings.npy"
CHUNKS_PATH        = "data/chunks.json"
# ──────────────────────────────────────────────────────────


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"📦 Loaded {len(chunks)} chunks, embeddings shape: {embeddings.shape}")
    return embeddings, chunks


def get_doc_type(source_text: str, chunk_id: int, total_chunks: int) -> str:
    """
    Determines which PDF a chunk came from based on its position.
    Technical regs were ingested first (chunks 0 to ~60% of total).
    Sporting regs follow after.

    TODO: A cleaner approach (Day 4+) is to store the source filename
    per chunk during ingestion so you don't need to estimate here.
    For now this approximation works because we know the ingestion order.
    """
    technical_ratio = 178 / (178 + 111)   # pages in technical vs total pages
    if chunk_id < int(total_chunks * technical_ratio):
        return "technical_regulations"
    return "sporting_regulations"


def create_collection(client: QdrantClient):
    """
    Creates a Qdrant collection — think of this as creating a table in a database.

    VectorParams defines:
    - size: how many dimensions each vector has (must match your embedding model)
    - distance: how similarity is measured. COSINE = same as what you did manually
      with sklearn. Qdrant does this internally now, indexed for speed.

    if_not_exists=True means re-running this script won't crash if collection exists.
    """
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        ),
        on_disk_payload=True    # store metadata on disk not RAM — better for large datasets
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")


def ingest(client: QdrantClient, embeddings: np.ndarray, chunks: list[dict]):
    """
    Uploads vectors + metadata to Qdrant in batches.

    Each item is a PointStruct with 3 fields:
    - id: unique integer identifier for this point
    - vector: the 1024-dim embedding as a list
    - payload: a dict of metadata you can filter on later

    The payload is the key upgrade from Day 1/2 — you can now query
    "give me chunks where doc_type = sporting_regulations" instead of
    searching everything blindly.

    Batch size of 100 means we upload 100 points per API call.
    Avoids memory issues and timeouts on large datasets.
    """
    total = len(chunks)
    batch_size = 100
    points_uploaded = 0

    for i in range(0, total, batch_size):
        batch_chunks     = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        points = [
            PointStruct(
                id=chunk["id"],
                vector=batch_embeddings[j].tolist(),   # numpy array → plain list
                payload={
                    "text":     chunk["text"],
                    "doc_type": get_doc_type(chunk["source"], chunk["id"], total),
                    "chunk_id": chunk["id"],
                    # TODO Day 4+: add "season": "2024", "article": "34.3" etc
                }
            )
            for j, chunk in enumerate(batch_chunks)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        points_uploaded += len(points)
        print(f"  Uploaded {points_uploaded}/{total} points...")

    print(f"✅ All {total} points ingested into Qdrant")


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)
    print(f"🔌 Connected to Qdrant at {QDRANT_URL}")

    # Delete collection if it already exists (clean re-run)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")

    embeddings, chunks = load_data()
    create_collection(client)
    ingest(client, embeddings, chunks)

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n📊 Collection info:")
    print(f"   Vectors count: {info.points_count}")
    print(f"   Status: {info.status}")
    print(f"\n🌐 View in dashboard: http://localhost:6333/dashboard")
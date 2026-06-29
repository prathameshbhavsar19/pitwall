# retrieve.py — Day 3
# Upgraded from numpy linear scan → Qdrant indexed search
# New feature: filter by doc_type (technical vs sporting regulations)

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import json

# ── CONFIG ────────────────────────────────────────────────
COLLECTION_NAME = "pitwall"
MODEL_NAME      = "BAAI/bge-large-en-v1.5"
QDRANT_URL      = "http://localhost:6333"
TOP_K           = 5
# ──────────────────────────────────────────────────────────

TEST_QUERIES = [
    ("What is the penalty for unsafe release in the pit lane?",  None),
    ("What is the maximum allowed front wing width?",            None),
    ("How many power unit components are allowed per season?",   None),
    ("What are the rules around tyre compounds at a race weekend?", None),
    ("What is the minimum car weight including the driver?",     None),
    ("What happens if a driver misses the weigh bridge?",        None),
    ("What are the DRS activation zone rules?",                  None),
    ("What is the fuel flow rate limit during a race?",          None),
    ("What defines a safety car period?",                        None),
    ("What are the pit lane speed limit rules?",                 None),
    # Filtered queries — search only one document
    ("What is the penalty for unsafe release?",   "sporting_regulations"),
    ("What is the maximum front wing width?",     "technical_regulations"),
]
# Each query is a tuple: (question, doc_type_filter or None)
# None = search everything, "sporting_regulations" or "technical_regulations" = filtered


def retrieve(query: str, model, client: QdrantClient, doc_type: str = None):
    """
    Replaces the numpy cosine similarity loop from Day 1/2.

    Key differences:
    1. Qdrant uses HNSW index — sub-millisecond search regardless of collection size
       vs your old linear scan which got slower as chunks grew
    2. Optional metadata filter — pass doc_type to search only one PDF's chunks
    3. Returns scored results directly — no manual argsort needed

    The filter works by checking the "doc_type" field in each point's payload.
    Only points where doc_type matches are considered in the search.
    """
    q_vec = model.encode([query], convert_to_numpy=True)[0].tolist()

    # Build filter only if doc_type is specified
    query_filter = None
    if doc_type:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type)
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        query_filter=query_filter,
        limit=TOP_K,
        with_payload=True
    ).points

    return results


if __name__ == "__main__":
    print(f"🔧 Loading model: {MODEL_NAME}")
    model  = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    print(f"🔌 Connected to Qdrant\n")

    for query, doc_filter in TEST_QUERIES:
        filter_label = f" [filter: {doc_filter}]" if doc_filter else " [no filter]"
        print(f"\n{'='*60}")
        print(f"🔍 {query}{filter_label}")
        print(f"{'='*60}")

        results = retrieve(query, model, client, doc_filter)

        for i, r in enumerate(results):
            print(f"\n  Rank {i+1} | Score: {round(r.score, 4)} | doc_type: {r.payload['doc_type']}")
            print(f"  {r.payload['text'][:250]}...")
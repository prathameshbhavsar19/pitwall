"""
Retrieval service — wraps Qdrant queries so the API layer never talks
to Qdrant directly.

Strategy: Hybrid retrieval
  - Dense search via Qdrant (bge-large embeddings, semantic similarity)
  - Sparse search via BM25 (keyword matching, built at startup)
  - Fused with Reciprocal Rank Fusion (RRF, k=60)
"""
import re
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI

# ── OpenAI client ─────────────────────────────────────────────────────
_openai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

GPT_DEPLOYMENT     = "gpt-4o"
GPT_MAX_TOKENS     = 512
COST_PER_1K_INPUT  = 0.0025   # Azure GPT-4o pricing
COST_PER_1K_OUTPUT = 0.010

# ── Qdrant + embedding config ─────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME  = "pitwall"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RRF_K            = 60

# Loaded once at module import — avoids reloading per request
if QDRANT_API_KEY:
    # Qdrant Cloud — URL-based connection
    _client = QdrantClient(
        url=f"https://{QDRANT_HOST}",
        api_key=QDRANT_API_KEY,
    )
else:
    # Local Qdrant — host:port connection
    _client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )
_model  = SentenceTransformer(EMBED_MODEL_NAME)


# ── Tokenizer ─────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """
    Lowercase, split on whitespace, strip leading/trailing punctuation.
    '54.3d)' → '54.3d'
    """
    return [re.sub(r'^\W+|\W+$', '', t) for t in text.lower().split() if t]


# ── BM25 index — built once at startup ───────────────────────────────
def _build_bm25_index():
    """
    Scroll every point in Qdrant, tokenize text payloads, build BM25 index.
    Returns (bm25, corpus_chunks).
    """
    all_chunks  = []
    next_offset = None

    while True:
        results, next_offset = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload or {}
            all_chunks.append({
                "id":       point.id,
                "text":     payload.get("text", ""),
                "doc_type": payload.get("doc_type"),
                "chunk_id": payload.get("chunk_id"),
            })
        if next_offset is None:
            break

    tokenized = [_tokenize(chunk["text"]) for chunk in all_chunks]
    bm25      = BM25Okapi(tokenized)
    return bm25, all_chunks


print("Building BM25 index from Qdrant...")
_bm25, _corpus = _build_bm25_index()
print(f"BM25 index ready — {len(_corpus)} chunks indexed.")


# ── Public helpers ────────────────────────────────────────────────────
def check_ready() -> bool:
    """Lightweight check that Qdrant is reachable and collection exists."""
    try:
        names = [c.name for c in _client.get_collections().collections]
        return COLLECTION_NAME in names
    except Exception:
        return False


# ── RRF merge ─────────────────────────────────────────────────────────
def _rrf_merge(
    dense_results: list[dict],
    bm25_results:  list[dict],
    top_k:         int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over two ranked lists.
    score = 1/(k + rank_dense) + 1/(k + rank_bm25)
    Absent chunks get rank = len(corpus) + 1 as penalty.
    """
    dense_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
    bm25_ranks  = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}
    all_ids     = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    penalty     = len(_corpus) + 1

    scores = {}
    for cid in all_ids:
        r_dense     = dense_ranks.get(cid, penalty)
        r_bm25      = bm25_ranks.get(cid, penalty)
        scores[cid] = 1 / (RRF_K + r_dense) + 1 / (RRF_K + r_bm25)

    payload_lookup = {r["chunk_id"]: r for r in dense_results + bm25_results}
    ranked         = sorted(all_ids, key=lambda cid: scores[cid], reverse=True)[:top_k]

    return [
        {**payload_lookup[cid], "score": round(scores[cid], 6)}
        for cid in ranked
    ]


# ── Search ────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 5, doc_type: str | None = None) -> list[dict]:
    """
    Hybrid search: dense (Qdrant) + sparse (BM25), fused with RRF.
    Returns list of dicts: {text, score, doc_type, chunk_id}.
    """
    # 1. Dense search
    vector       = _model.encode(query).tolist()
    query_filter = None
    if doc_type is not None:
        query_filter = Filter(
            must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))]
        )

    dense_response = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k * 3,
    )
    dense_results = []
    for point in dense_response.points:
        payload = point.payload or {}
        dense_results.append({
            "text":     payload.get("text", ""),
            "score":    point.score,
            "doc_type": payload.get("doc_type"),
            "chunk_id": payload.get("chunk_id"),
        })

    # 2. BM25 search
    tokens        = _tokenize(query)
    bm25_scores   = _bm25.get_scores(tokens)
    scored_corpus = [
        {**chunk, "score": bm25_scores[i]}
        for i, chunk in enumerate(_corpus)
        if doc_type is None or chunk["doc_type"] == doc_type
    ]
    scored_corpus.sort(key=lambda x: x["score"], reverse=True)
    bm25_results = scored_corpus[: top_k * 3]

    # 3. RRF fusion
    return _rrf_merge(dense_results, bm25_results, top_k)


# ── Generation ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PitWall, an expert assistant on Formula 1 regulations.
Answer the user's question using ONLY the context chunks provided.
Answer in 2-3 sentences, directly and precisely addressing the question asked.
Cite which regulation (technical or sporting) your answer comes from.
If the context does not contain enough information to answer, say so clearly.
Never make up rules or penalties that are not in the provided context."""


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Takes retrieved chunks, calls GPT-4o, returns answer + token usage + cost.
    Returns { answer, prompt_tokens, completion_tokens, total_tokens, cost_usd }
    """
    context = "\n\n".join(
        f"[{i+1}] ({c['doc_type']}) {c['text']}"
        for i, c in enumerate(chunks)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    response = _openai.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=messages,
        max_tokens=GPT_MAX_TOKENS,
        temperature=0.1,
    )

    usage = response.usage
    cost  = (
        (usage.prompt_tokens     / 1000) * COST_PER_1K_INPUT +
        (usage.completion_tokens / 1000) * COST_PER_1K_OUTPUT
    )

    return {
        "answer":            response.choices[0].message.content,
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens":      usage.total_tokens,
        "cost_usd":          round(cost, 6),
    }
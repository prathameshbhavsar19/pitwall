# PitWall 🏎

> A production RAG API over 2024 F1 FIA regulations — built to demonstrate enterprise-grade retrieval, evaluation, observability, and deployment practices.

**Live demo (deployed):** `https://pitwall-api.proudocean-4691f2bb.westus2.azurecontainerapps.io/docs`  
**Stack:** FastAPI · Qdrant Cloud · bge-large · BM25 · GPT-4o · Langfuse · GitHub Actions · Azure Container Apps

---

## What it does

PitWall answers natural language questions about F1 sporting and technical regulations using a hybrid RAG pipeline. Ask "What is the penalty for unsafe release during a race?" and get a precise, cited answer generated from the actual FIA regulation text — not a hallucination.

```bash
curl -X POST https://pitwall-api.proudocean-4691f2bb.westus2.azurecontainerapps.io/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the penalty for unsafe release during a race?", "top_k": 3}'
```

```json
{
  "generated_answer": "During a race or sprint session, an unsafe release results in a penalty in accordance with Article 54.3d of the Sporting Regulations. If the driver continues driving knowing the car was released unsafely, an additional penalty is imposed. (Sporting Regulations [1], [2])",
  "latency_ms": 4701,
  "tokens_used": 833,
  "cost_usd": 0.002847
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /query
┌─────────────────────▼───────────────────────────────────────┐
│               FastAPI (Azure Container Apps)                │
│                                                             │
│  1. Embed query        bge-large-en-v1.5 (1024-dim)        │
│  2. Dense search   →   Qdrant Cloud (HNSW, cosine)         │
│  3. Sparse search  →   BM25 (in-memory inverted index)     │
│  4. RRF fusion     →   Reciprocal Rank Fusion (k=60)       │
│  5. Generate       →   Azure GPT-4o (temp=0.1)             │
│  6. Trace          →   Langfuse (cost + latency)           │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
┌─────────▼────────┐  ┌────────▼───────┐  ┌────────▼────────┐
│   Qdrant Cloud   │  │  Azure OpenAI  │  │ Langfuse Cloud  │
│  1081 chunks     │  │    GPT-4o      │  │  Observability  │
│  1024-dim HNSW   │  │  gpt-4o-mini   │  │  Cost tracking  │
└──────────────────┘  └────────────────┘  └─────────────────┘
```

---

## Key technical decisions

### Why hybrid retrieval (dense + BM25 + RRF)?

Dense-only retrieval with bge-large is strong on semantic similarity but weak on exact matches. "Article 54.3d" has no semantic meaning to an embedding model — it's an opaque string. BM25's inverted index finds exact article references instantly. RRF fuses both ranked lists without needing to normalise scores (they're on incompatible scales). Result: the best of both worlds on every query.

The BM25 index is built at startup by scrolling all 1081 chunks from Qdrant and tokenising them with a custom `_tokenize()` function that strips trailing punctuation — a necessary fix for PDF extraction artifacts like `"54.3d)"` becoming `"54.3d"` in the index.

### Why sentence-boundary chunking (not semantic/paragraph)?

Benchmarked three strategies in Day 2: fixed (500/50), sentence boundary (5 sentences), paragraph. Paragraph chunking scored **0.5938** vs sentence chunking's **0.7242** — worst of the three. FIA regulations pack multiple sub-clauses into single paragraphs, creating diluted embeddings that match nothing well. Sentence chunking preserves sub-clause granularity. The benchmark result drove the decision, not the documentation.

### Why bge-large over MiniLM?

Benchmarked both on the same 10 queries. bge-large improved scores by **+0.13** — more impact than any chunking strategy change. The model upgrade was 13x more impactful than switching chunking strategies. Budget your optimisation effort accordingly.

### Why top_k=3 for generation?

Ablation showed top_k=3 outperforms top_k=5 on Context Precision (0.84 vs 0.81) and Faithfulness (0.87 vs 0.85). Fewer chunks means tighter context, more traceable claims, higher faithfulness. For a regulations bot where wrong answers mean wrong penalties, faithfulness matters more than completeness.

### Why not semantic chunking?

Day 2 benchmark result: paragraph/semantic chunking scored 0.59, worse than fixed chunking. FIA regulations are dense legal text where paragraphs contain unrelated sub-clauses — semantic chunkers bundle them into diluted-embedding chunks that match nothing. Sentence boundary respects sub-clause granularity.

---

## RAGAS evaluation results

Evaluated over 20 regulation QA pairs using gpt-4o-mini as judge. Hybrid retrieval + LLM generation pipeline:

| Metric | Score | CI Gate Threshold | Status |
|---|---|---|---|
| Context Precision | 0.8417 | ≥ 0.70 | ✅ PASS |
| Faithfulness | 0.8667 | ≥ 0.75 | ✅ PASS |
| Answer Relevancy | 0.7482 | ≥ 0.70 | ✅ PASS |

**Observed variance:** ±0.05 across runs on a 20-question eval set (probabilistic judge model). CI gate thresholds set with headroom to avoid flaky PR failures while still catching genuine regressions.

### RAGAS progression across days

| Day | Strategy | Context Precision | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| Day 6 | Dense only, raw chunks | 0.8078 | 0.9579 | 0.5260 |
| Day 7 | Hybrid RRF, raw chunks | 0.7706 | 0.9544 | 0.4673 |
| Day 8 | Hybrid RRF + GPT-4o generation | 0.8417 | 0.8667 | 0.7482 |

Answer Relevancy jumped from 0.47 to 0.75 purely from switching the RAGAS response field from raw retrieved chunks to GPT-4o generated answers. The metric was measuring the wrong thing before Day 8.

---

## Load test results

Tested with locust at 50 and 150 concurrent users:

| Metric | 50 users | 150 users |
|---|---|---|
| Sustained RPS | 9.4 | 12.1 |
| Failure rate | 0% | 0.057% |
| p50 latency | 3,000ms | 12,000ms |
| p95 latency | 7,000ms | 16,000ms |
| p99 latency | — | 20,000ms |
| Single-request baseline | — | 1,259ms |

**Bottleneck identified:** Azure OpenAI GPT-4o deployment throughput (~12 RPS ceiling). FastAPI, Qdrant, and BM25 layers showed zero failures under load. The system queues rather than rejects — every request succeeds but latency grows as requests stack up behind Azure's processing capacity. To scale: upgrade to a higher-TPM Azure deployment, not more API servers.

---

## CI/CD

GitHub Actions pipeline with two jobs on every pull request:

```
pytest          → spins up Qdrant, ingests corpus, runs 5 unit tests
ragas-eval      → starts API, runs full 20-question RAGAS eval,
                   blocks PR if faithfulness < 0.75 or precision < 0.70
```

Embedding cache keyed on `hashFiles('data/*.pdf')` — skips the ~5-minute embedding step when regulations haven't changed.

→ [View CI runs](https://github.com/prathameshbhavsar19/pitwall/actions)

---

## Observability

Every `/query` request traced in Langfuse with:
- Token counts (prompt + completion)
- Cost in USD per request (~$0.00225 average)
- Latency breakdown
- Generated answer + retrieved chunks

Cost tracking uses `as_type="generation"` with structured `usage_details` and `cost_details` — not generic output metadata, which Langfuse ignores for cost computation.

---

## Running locally

**Prerequisites:** Python 3.11, Docker, Azure OpenAI endpoint, Qdrant Cloud cluster

```bash
git clone https://github.com/prathameshbhavsar19/pitwall
cd pitwall
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your credentials
```

**Ingest corpus into Qdrant:**
```bash
python ingest/ingest.py       # chunk PDFs
python ingest/embed.py        # generate embeddings
python ingest/qdrant_ingest.py  # upload to Qdrant
```

**Start the API:**
```bash
python -m uvicorn api.main:app
# → http://127.0.0.1:8000/docs
```

**Run RAGAS eval:**
```bash
python eval/ragas_eval.py
```

**Run load test:**
```bash
locust -f locustfile.py --host http://127.0.0.1:8000
# → http://localhost:8089
```

---

## Project structure

```
pitwall/
├── api/
│   ├── main.py               # FastAPI app, rate limiting, Langfuse tracing
│   ├── models.py             # Pydantic request/response models
│   └── retrieval_service.py  # Dense + BM25 + RRF hybrid search, GPT-4o generation
├── ingest/
│   ├── ingest.py             # PDF → sentence-boundary chunks
│   ├── embed.py              # bge-large embeddings
│   └── qdrant_ingest.py      # Upload to Qdrant Cloud
├── eval/
│   ├── qa_pairs.json         # 20 regulation QA pairs (ground truth)
│   ├── ragas_eval.py         # RAGAS evaluation pipeline
│   └── ragas_results.json    # Latest eval scores
├── dashboard/
│   └── app.py                # Streamlit ops dashboard
├── tests/
│   └── test_api.py           # 5 pytest tests
├── .github/workflows/
│   └── eval.yml              # CI gate: pytest + RAGAS eval
├── locustfile.py             # Load test
├── Dockerfile                # linux/amd64, bge-large baked in
└── .env.example              # Required environment variables
```

---

## Environment variables

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | API version (e.g. `2024-12-01-preview`) |
| `QDRANT_HOST` | Qdrant Cloud cluster hostname |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host (default: `https://cloud.langfuse.com`) |

---

## Deployment

Containerised with Docker (`linux/amd64`, bge-large baked into image for fast cold start) and deployed to Azure Container Apps with Qdrant Cloud as the vector store.

```bash
# Build for Azure (ARM Mac → linux/amd64)
docker buildx build --platform linux/amd64 \
  -t pitwallregistry.azurecr.io/pitwall-api:latest --push .

# Deploy to Azure Container Apps
az containerapp create \
  --name pitwall-api \
  --resource-group pitwall-rg \
  --environment pitwall-env \
  --image pitwallregistry.azurecr.io/pitwall-api:latest \
  --cpu 2 --memory 4Gi \
  --ingress external --target-port 8000
```

---

## What I'd do differently at scale

- **Reranking:** add a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM`) between retrieval and generation to reorder chunks before stuffing into the prompt. Addresses the "lost in the middle" problem at higher top_k values.
- **Semantic chunking:** not beneficial here (paragraph chunking scored 0.59 vs 0.72 for sentence boundary on this corpus), but would revisit for prose-heavy documents.
- **RAGAS at scale:** current 20-question eval has ±0.05 variance. 500+ questions with human-annotated ground truth would make the eval gate reliable enough to set tighter thresholds.
- **Azure OpenAI throughput:** ~12 RPS ceiling from a single deployment. A paid higher-TPM tier or multiple deployments behind a load balancer would extend this proportionally.
- **Streaming:** add SSE streaming to `/query` so users see tokens appear immediately rather than waiting 2-4 seconds for the full generated answer.

---

*Portfolio: [prathameshbhavsar19.github.io](https://prathameshbhavsar19.github.io)*
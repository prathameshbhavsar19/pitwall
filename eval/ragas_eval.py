"""
Day 6 — RAGAS evaluation script (ragas 0.4.x API).

Measures 3 metrics on your current retrieval pipeline:
  - Context Precision   : are retrieved chunks relevant to the question?
  - Faithfulness        : is the answer grounded in the retrieved chunks?
  - Answer Relevancy    : does the answer actually address the question?

Run from pitwall/ root:
    python eval/ragas_eval.py

Requires:
    AZURE_OPENAI_* keys in .env
    Qdrant running with pitwall collection
    API running: python -m uvicorn api.main:app
"""

import json
import os
import time
import requests
from dotenv import load_dotenv
from ragas.run_config import RunConfig


load_dotenv()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    _LLMContextPrecisionWithReference as LLMContextPrecision,
    _Faithfulness as Faithfulness,
    _ResponseRelevancy as ResponseRelevancy,
)
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
API_URL      = "http://127.0.0.1:8000/query"
QA_PATH      = "eval/qa_pairs.json"
TOP_K        = 3
RESULTS_PATH = "eval/ragas_results.json"

# ── Local embeddings wrapper ──────────────────────────────────────────────────
class LocalEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

# ── Setup LLM and embeddings for ragas ───────────────────────────────────────
llm = LangchainLLMWrapper(AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4o-mini",
    temperature=0,
))

embeddings = LocalEmbeddings("BAAI/bge-large-en-v1.5")

# ── Instantiate metrics ───────────────────────────────────────────────────────
context_precision = LLMContextPrecision(llm=llm)
faithfulness      = Faithfulness(llm=llm)
answer_relevancy  = ResponseRelevancy(llm=llm, embeddings=embeddings)

# ── Step 1: Load Q&A pairs ────────────────────────────────────────────────────
with open(QA_PATH) as f:
    qa_pairs = json.load(f)

print(f"Loaded {len(qa_pairs)} Q&A pairs")

# ── Step 2: Run retrieval for each question ───────────────────────────────────
print("\nRunning retrieval for each question...")

samples = []
for i, pair in enumerate(qa_pairs):
    question     = pair["question"]
    ground_truth = pair["ground_truth"]

    try:
        response = requests.post(
            API_URL,
            json={"query": question, "top_k": TOP_K},
            timeout=30,
        )
        response.raise_for_status()
        data       = response.json()
        contexts   = [r["text"] for r in data["results"]]
        top_answer = data.get("generated_answer") or (contexts[0] if contexts else "No results found.")
        top_score  = data["results"][0]["score"] if data["results"] else 0
    except Exception as e:
        print(f"  [{i+1:02d}] FAILED: {e}")
        contexts   = []
        top_answer = "Retrieval failed."
        top_score  = 0.0

    samples.append(
        SingleTurnSample(
            user_input=question,
            response=top_answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
    )

    print(f"  [{i+1:02d}] {question[:60]}... score={top_score:.3f}")
    time.sleep(7)  # stay under 10 req/min rate limit

# ── Step 3: Build dataset and evaluate ───────────────────────────────────────
dataset = EvaluationDataset(samples=samples)
print(f"\nDataset built: {len(samples)} samples")
print("Running RAGAS evaluation (LLM via Azure, embeddings local)...")

results = evaluate(
    dataset=dataset,
    metrics=[context_precision, faithfulness, answer_relevancy],
    run_config=RunConfig(
        timeout=120,        # give each judge call more time
        max_workers=4,      # reduce concurrency to avoid rate limits
    ),
)

# ── Step 4: Print results ─────────────────────────────────────────────────────
df = results.to_pandas()
print("\nAvailable columns:", df.columns.tolist())

col_cp = [c for c in df.columns if "precision" in c.lower()][0]
col_ff = [c for c in df.columns if "faithful" in c.lower()][0]
col_ar = [c for c in df.columns if "relevan" in c.lower()][0]

avg_cp = df[col_cp].mean()
avg_ff = df[col_ff].mean()
avg_ar = df[col_ar].mean()

print("\n" + "="*55)
print("RAGAS RESULTS (hybrid retrieval + LLM generation)")
print("="*55)
print(f"  Context Precision : {avg_cp:.4f}  (target ≥ 0.70)")
print(f"  Faithfulness      : {avg_ff:.4f}  (target ≥ 0.75)")
print(f"  Answer Relevancy  : {avg_ar:.4f}  (target ≥ 0.80)")
print("="*55)

passed = sum([avg_cp >= 0.70, avg_ff >= 0.75, avg_ar >= 0.80])
print(f"\n{passed}/3 metrics hit target.")

# ── Step 5: Save results ──────────────────────────────────────────────────────
output = {
    "context_precision": round(float(avg_cp), 4),
    "faithfulness":      round(float(avg_ff), 4),
    "answer_relevancy":  round(float(avg_ar), 4) if str(avg_ar) != "nan" else None,
    "num_questions":     len(qa_pairs),
    "top_k":             TOP_K,
    "model":             "BAAI/bge-large-en-v1.5",
    "retrieval": "hybrid-rrf",
    "generation": "gpt-4o (Azure)",
    "judge_model":       "gpt-4o-mini (Azure)",
    "embed_model":       "BAAI/bge-large-en-v1.5 (local)",
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print("These are your baseline — compare after hybrid retrieval on Day 7.")
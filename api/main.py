from dotenv import load_dotenv
load_dotenv()

import time
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .models import QueryRequest, QueryResponse, QueryResult, ErrorResponse
from . import retrieval_service

# --- Langfuse setup ---------------------------------------------------
try:
    from langfuse import Langfuse
    langfuse = Langfuse()
    langfuse.auth_check()
    LANGFUSE_ENABLED = True
except Exception as e:
    langfuse = None
    LANGFUSE_ENABLED = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pitwall.api")
if LANGFUSE_ENABLED:
    logger.info("Langfuse enabled")
else:
    logger.warning("Langfuse disabled — traces will not be sent")

# --- Rate limiter setup -----------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="PitWall API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if retrieval_service.check_ready():
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="not_ready",
            detail="Qdrant unreachable or 'pitwall' collection missing",
        ).model_dump(),
    )


@app.post("/query", response_model=QueryResponse, responses={
    400: {"model": ErrorResponse},
    429: {"description": "Rate limit exceeded"},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
})
@limiter.limit("10/minute")
def query(request: Request, body: QueryRequest):
    start = time.perf_counter()

    # --- Input validation ------------------------------------------------
    valid_doc_types = {"technical_regulations", "sporting_regulations", None}
    if body.doc_type not in valid_doc_types:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="invalid_doc_type",
                detail=f"doc_type must be one of: technical_regulations, sporting_regulations (got '{body.doc_type}')",
            ).model_dump(),
        )

    # --- Retrieval -------------------------------------------------------
    try:
        raw_results = retrieval_service.search(
            query=body.query,
            top_k=body.top_k,
            doc_type=body.doc_type,
        )
    except ConnectionError as e:
        logger.error(f"Qdrant connection error: {e}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="retrieval_unavailable",
                detail="Could not connect to vector database",
            ).model_dump(),
        )
    except Exception as e:
        logger.exception("Unexpected error during retrieval")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred during retrieval",
            ).model_dump(),
        )

    # --- Generation ------------------------------------------------------
    gen = None
    try:
        gen = retrieval_service.generate(
            query=body.query,
            chunks=raw_results,
        )
    except Exception:
        logger.exception("Generation failed — returning retrieval results only")

    latency_ms = (time.perf_counter() - start) * 1000
    results    = [QueryResult(**r) for r in raw_results]

    response = QueryResponse(
        query=body.query,
        results=results,
        generated_answer=gen["answer"]      if gen else None,
        tokens_used=gen["total_tokens"]     if gen else None,
        cost_usd=gen["cost_usd"]            if gen else None,
        latency_ms=latency_ms,
    )

    # --- Langfuse tracing ------------------------------------------------
    if LANGFUSE_ENABLED:
        try:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="pitwall-query",
                model="gpt-4o",
            ):
                langfuse.update_current_generation(
                    input={
                        "query":    body.query,
                        "doc_type": body.doc_type,
                        "top_k":    body.top_k,
                    },
                    output={
                        "generated_answer": gen["answer"] if gen else None,
                        "num_results":      len(results),
                        "latency_ms":       latency_ms,
                        "top_score":        results[0].score if results else None,
                    },
                    usage_details={
                        "input":  gen["prompt_tokens"]     if gen else 0,
                        "output": gen["completion_tokens"] if gen else 0,
                        "total":  gen["total_tokens"]      if gen else 0,
                    } if gen else None,
                    cost_details={
                        "input":  round(gen["cost_usd"] * gen["prompt_tokens"] / gen["total_tokens"], 6),
                        "output": round(gen["cost_usd"] * gen["completion_tokens"] / gen["total_tokens"], 6),
                        "total":  gen["cost_usd"],
                    } if gen else None,
                )
            langfuse.flush()
        except Exception:
            logger.warning("Langfuse trace logging failed", exc_info=True)

    return response
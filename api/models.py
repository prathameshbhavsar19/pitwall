from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query text")
    doc_type: str | None = Field(
        default=None,
        description="Optional filter: 'technical_regulations' or 'sporting_regulations'",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class QueryResult(BaseModel):
    text: str
    score: float
    doc_type: str | None = None
    chunk_id: int | None = None


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResult]
    generated_answer: str | None = None 
    latency_ms: float
    tokens_used: int | None = None        
    cost_usd: float | None = None         


class ErrorResponse(BaseModel):
    error: str
    detail: str
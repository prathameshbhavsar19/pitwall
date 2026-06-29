"""
Day 5 pytest suite — 5 tests covering happy path + all error cases.

Run from pitwall/ root:
    pytest tests/test_api.py -v

Tests 1, 5 hit real Qdrant (need Docker running).
Tests 2, 3, 4 mock retrieval_service — no Qdrant needed.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# TestClient lets us make HTTP requests to the FastAPI app
# without running a real server — it calls the app directly in-process.
from api.main import app

client = TestClient(app)

# --- Fake results used by mocked tests --------------------------------
FAKE_RESULTS = [
    {
        "text": "Cars must not be released in an unsafe condition.",
        "score": 0.72,
        "doc_type": "sporting_regulations",
        "chunk_id": 860,
    }
]


# --- Test 1: Happy path -----------------------------------------------
def test_query_returns_results():
    """
    Real end-to-end test — hits actual Qdrant.
    Confirms the response shape matches QueryResponse exactly.
    Needs: Docker + Qdrant running with pitwall collection.
    """
    response = client.post("/query", json={"query": "unsafe release", "top_k": 3})

    assert response.status_code == 200
    data = response.json()

    # Check top-level keys exist
    assert "query" in data
    assert "results" in data
    assert "latency_ms" in data

    # Check query is echoed back
    assert data["query"] == "unsafe release"

    # Check results is a list with the right shape
    assert isinstance(data["results"], list)
    assert len(data["results"]) <= 3  # respects top_k

    if data["results"]:
        result = data["results"][0]
        assert "text" in result
        assert "score" in result
        assert isinstance(result["score"], float)


# --- Test 2: Empty query → 422 ----------------------------------------
def test_empty_query_returns_422():
    """
    Pydantic's min_length=1 on QueryRequest.query should reject
    an empty string before our code even runs.
    No Qdrant needed — validation fires before retrieval.
    """
    with patch("api.retrieval_service.search", return_value=FAKE_RESULTS):
        response = client.post("/query", json={"query": ""})

    assert response.status_code == 422


# --- Test 3: Invalid doc_type → 400 -----------------------------------
def test_invalid_doc_type_returns_400():
    """
    Our whitelist check (valid_doc_types) should catch any doc_type
    that isn't 'technical_regulations' or 'sporting_regulations'.
    Returns our structured ErrorResponse, not a Pydantic 422.
    """
    with patch("api.retrieval_service.search", return_value=FAKE_RESULTS):
        response = client.post(
            "/query",
            json={"query": "unsafe release", "doc_type": "made_up_type"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "invalid_doc_type"
    assert "made_up_type" in data["detail"]


# --- Test 4: top_k out of range → 422 ---------------------------------
def test_top_k_out_of_range_returns_422():
    """
    QueryRequest.top_k has ge=1, le=20.
    Anything outside that range should be caught by Pydantic (422),
    not by our code — so no Qdrant needed.
    """
    with patch("api.retrieval_service.search", return_value=FAKE_RESULTS):
        response = client.post(
            "/query",
            json={"query": "unsafe release", "top_k": 99},
        )

    assert response.status_code == 422


# --- Test 5: /ready returns 200 when Qdrant is up ---------------------
def test_ready_returns_200():
    """
    Real test — checks /ready can actually reach Qdrant and find
    the pitwall collection.
    Needs: Docker + Qdrant running with pitwall collection.
    """
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
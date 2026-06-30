"""
PitWall Ops Dashboard
Run: streamlit run dashboard/app.py
"""

import os, json, base64, time
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
RAGAS_RESULTS_PATH  = "eval/ragas_results.json"


def _auth_header():
    token = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}"}


@st.cache_data(ttl=300)
def fetch_observations(days: int = 7):
    """Fetch all observations for the last N days, filter to pitwall-query client-side."""
    now   = datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).isoformat()

    rows, cursor = [], None
    for _ in range(20):
        params = {
            "fromStartTime": since,
            "fields":        "core,basic,usage",
            "limit":         100,
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(
            f"{LANGFUSE_HOST}/api/public/v2/observations",
            headers=_auth_header(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("data", [])
        rows += [o for o in batch if o.get("name") == "pitwall-query"]
        cursor = data.get("meta", {}).get("cursor")
        if not cursor or not batch:
            break
        time.sleep(0.3)

    return rows


def load_ragas():
    try:
        with open(RAGAS_RESULTS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def parse_latency_ms(o: dict) -> float | None:
    """
    Try every known field shape Langfuse v2 might return for duration.
    Returns milliseconds, or None if unparseable.
    """
    # Shape 1 — latency field in ms (some Langfuse versions)
    if o.get("latency") is not None:
        return float(o["latency"]) * 1000  # latency is in seconds

    # Shape 2 — startTime / endTime ISO strings
    raw_start = o.get("startTime") or o.get("start_time")
    raw_end   = o.get("endTime")   or o.get("end_time")
    if raw_start and raw_end:
        try:
            s = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
            e = datetime.fromisoformat(raw_end.replace("Z",   "+00:00"))
            diff = (e - s).total_seconds() * 1000
            if diff > 0:
                return diff
        except Exception:
            pass

    # Shape 3 — duration in ms directly
    if o.get("duration") is not None:
        return float(o["duration"])

    return None


# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="PitWall Ops", page_icon="🏎", layout="wide")
st.title("PitWall — Ops Dashboard")
st.caption(
    f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}  "
    f"·  data cached 5 min"
)

days = st.sidebar.slider("Lookback window (days)", 1, 30, 7)
st.sidebar.divider()
if st.sidebar.button("Force refresh"):
    st.cache_data.clear()
    st.rerun()

# ── Fetch data ────────────────────────────────────────────────────────
with st.spinner("Fetching traces from Langfuse..."):
    try:
        obs   = fetch_observations(days)
        lf_ok = True
    except Exception as e:
        st.warning(f"Langfuse fetch failed: {e}")
        obs, lf_ok = [], False

ragas = load_ragas()

# Debug panel — shows in sidebar when obs exist, helps confirm field shapes
if obs and st.sidebar.checkbox("Show raw field debug", value=False):
    sample = obs[0]
    st.sidebar.write("**Fields returned:**", list(sample.keys()))
    st.sidebar.write("**startTime:**",  sample.get("startTime"))
    st.sidebar.write("**endTime:**",    sample.get("endTime"))
    st.sidebar.write("**latency:**",    sample.get("latency"))
    st.sidebar.write("**duration:**",   sample.get("duration"))
    st.sidebar.write("**totalCost:**",  sample.get("totalCost"))

# ── Derived metrics ───────────────────────────────────────────────────
total_requests = len(obs)
errors         = [o for o in obs if (o.get("level") or "").upper() == "ERROR"]
failure_rate   = (len(errors) / total_requests * 100) if total_requests else 0

costs    = [o.get("totalCost") or 0 for o in obs]
avg_cost = sum(costs) / len(costs) if costs else 0

latencies = sorted([
    ms for o in obs
    for ms in [parse_latency_ms(o)]
    if ms is not None and ms > 0
])
p50 = latencies[int(len(latencies) * 0.50)] if latencies else None

# ── Row 1 — Health overview ───────────────────────────────────────────
st.subheader("Health overview")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total requests",       total_requests)
c2.metric("Avg cost / query",     f"${avg_cost:.5f}" if avg_cost else "—")
c3.metric("Median latency (p50)", f"{p50/1000:.2f}s" if p50 else "—")
c4.metric("Failure rate",         f"{failure_rate:.2f}%", delta_color="inverse")

# ── Row 2 — RAGAS quality ─────────────────────────────────────────────
st.divider()
st.subheader("Retrieval quality (RAGAS)")

if ragas:
    q1, q2, q3 = st.columns(3)
    cp = ragas.get("context_precision", 0)
    ff = ragas.get("faithfulness",      0)
    ar = ragas.get("answer_relevancy",  0)

    q1.metric(
        "Context Precision", f"{cp:.4f}",
        delta=f"{'PASS' if cp >= 0.70 else 'FAIL'} (>=0.70)",
        delta_color="normal" if cp >= 0.70 else "inverse",
    )
    q2.metric(
        "Faithfulness", f"{ff:.4f}",
        delta=f"{'PASS' if ff >= 0.75 else 'FAIL'} (>=0.75)",
        delta_color="normal" if ff >= 0.75 else "inverse",
    )
    q3.metric(
        "Answer Relevancy", f"{ar:.4f}",
        delta=f"{'PASS' if ar >= 0.70 else 'FAIL'} (>=0.70)",
        delta_color="normal" if ar >= 0.70 else "inverse",
    )
    st.caption(
        f"From `{RAGAS_RESULTS_PATH}` · "
        f"retrieval: {ragas.get('retrieval','—')} · "
        f"judge: {ragas.get('judge_model','—')} · "
        f"n={ragas.get('num_questions','—')}"
    )
else:
    st.info("No RAGAS results found. Run `python eval/ragas_eval.py` first.")

# ── Row 3 — Requests and cost over time ───────────────────────────────
st.divider()
st.subheader("Requests and cost over time")

if obs:
    df_rows = []
    for o in obs:
        try:
            ts   = o.get("startTime") or o.get("start_time") or ""
            hour = datetime.fromisoformat(
                ts.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:00")
            df_rows.append({"hour": hour, "cost": o.get("totalCost") or 0})
        except Exception:
            pass

    if df_rows:
        df  = pd.DataFrame(df_rows)
        agg = (
            df.groupby("hour")
            .agg(requests=("cost", "count"), total_cost=("cost", "sum"))
            .reset_index()
            .sort_values("hour")
        )
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Requests per hour**")
            st.bar_chart(agg.set_index("hour")["requests"])
        with col_r:
            st.markdown("**Cost per hour ($)**")
            st.bar_chart(agg.set_index("hour")["total_cost"])
else:
    st.info(
        "No Langfuse traces in this window. "
        "Fire a query at http://127.0.0.1:8000/query then hit Force refresh."
    )

# ── Row 4 — Live latency distribution ────────────────────────────────
if len(latencies) >= 5:
    st.divider()
    st.subheader("Latency distribution (live traces)")
    n = len(latencies)

    l1, l2, l3 = st.columns(3)
    l1.metric("p50",
              f"{latencies[int(n * 0.50)] / 1000:.2f}s")
    l2.metric("p95",
              f"{latencies[int(n * 0.95)] / 1000:.2f}s" if n >= 20  else "need 20+ traces")
    l3.metric("p99",
              f"{latencies[int(n * 0.99)] / 1000:.2f}s" if n >= 100 else "need 100+ traces")

    lat_df = pd.DataFrame({"latency_s": [ms / 1000 for ms in latencies]})
    st.bar_chart(lat_df["latency_s"], height=180)
    st.caption(f"Based on {n} traces · sorted fastest → slowest")

elif lf_ok and total_requests > 0:
    st.divider()
    st.info(
        f"Latency data not yet parseable from Langfuse for these {total_requests} traces. "
        "Enable 'Show raw field debug' in the sidebar to inspect the field names returned."
    )

# ── Row 5 — Load test reference ───────────────────────────────────────
st.divider()
st.subheader("Load test results (locust — 150 concurrent users)")

lt1, lt2, lt3, lt4, lt5 = st.columns(5)
lt1.metric("Total requests", "3,512")
lt2.metric("Failure rate",   "0.057%")
lt3.metric("p50 latency",    "12.0s")
lt4.metric("p95 latency",    "16.0s")
lt5.metric("p99 latency",    "20.0s")

st.caption(
    "Bottleneck: Azure OpenAI GPT-4o throughput ceiling (~12 RPS sustained). "
    "FastAPI + Qdrant + BM25 showed zero failures under load."
)
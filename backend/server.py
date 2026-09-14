"""CPU tier: the public API. Owns everything that is not a forward pass.

    uv run uvicorn backend.server:app --port 8000

Right now it validates a request and forwards it to the GPU tier at MODEL_URL.
Auth, sessions, conversation history, rate limits and billing land here rather
than in model/server.py, so the GPU tier stays swappable -- for vLLM, for
Bedrock, for a second model -- without any of that moving with it.

No streaming yet: this holds the request open and returns the whole string.
"""

import asyncio
import math
import os
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8080")
ORIGINS = ["http://localhost:3000"]
TIMEOUT_S = 300
BENCH_TIMEOUT_S = 1800     # 16 prompts through MAX_BATCH rows takes minutes

BENCH_PROMPTS = [
    "how to make pizza?",
    "what is 2+2?",
    "who are you?",
    "name a color",
    "explain recursion",
    "capital of France?",
    "name a fruit",
    "what is 5*5?",
    "how does a KV cache work?",
    "name an animal",
    "what is 10-7?",
    "write a haiku about GPUs",
    "what is the boiling point of water?",
    "name a programming language",
    "say hi",
    "name a country",
]

# One pooled client for the process
client = httpx.AsyncClient(base_url=MODEL_URL, timeout=TIMEOUT_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.aclose()


# RED at the public edge. The model tier reports the engine's own view; this
# is what a client actually experienced, proxy hop included.
EDGE = Histogram("socrates_edge_seconds", "end to end at the API tier",
                 buckets=(.5, 1, 2, 5, 10, 30, 60, 120, 300))
EDGE_REQUESTS = Counter("socrates_edge_requests_total", "api calls",
                        ["route", "status"])

app = FastAPI(title="socrates-backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    # auth, rate limit and history load go here, before the prompt is assembled
    started = time.perf_counter()
    try:
        r = await client.post("/generate", json={"prompt": req.prompt})
    except httpx.RequestError as e:
        EDGE_REQUESTS.labels("chat", "502").inc()
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    if r.status_code != 200:
        # pass the model tier's own 413/429/504 through rather than masking it
        EDGE_REQUESTS.labels("chat", str(r.status_code)).inc()
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise HTTPException(status_code=r.status_code, detail=detail)

    EDGE.observe(time.perf_counter() - started)
    EDGE_REQUESTS.labels("chat", "200").inc()

    body = r.json()
    return {
        "response": body["response"], 
        "timing": body.get("timing", {}),
        "output_tokens": body.get("output_tokens")
    }


@app.get("/health")
async def health() -> dict:
    """Reports the downstream too: this tier is useless without it."""
    try:
        r = await client.get("/health", timeout=2.0)
        return {"status": "ok", "model": r.json()}
    except httpx.RequestError:
        return {"status": "degraded", "model": None}


def _pct(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile; numpy is not a dependency of this tier."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _spread(xs: list[float]) -> dict[str, float]:
    return {f"p{p}": _pct(xs, p) for p in (50, 95, 99)}


@app.post("/api/benchmark")
async def benchmark() -> dict:
    """Fire every prompt at once and report what the scheduler did with them.

    Concurrency is the point: sequential requests would never fill a batch, so
    queue_s would be zero and occupancy would be 1/MAX_BATCH throughout.
    """
    try:
        meta = (await client.get("/health", timeout=10.0)).json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    t0 = time.perf_counter()

    async def one(i: int, prompt: str) -> dict:
        sent = time.perf_counter() - t0
        r = await client.post("/generate", json={"prompt": prompt}, timeout=BENCH_TIMEOUT_S)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        b = r.json()
        t = b["timing"]
        return {
            "i": i,
            "prompt": prompt,
            "prompt_tokens": b["prompt_tokens"],
            "output_tokens": b["output_tokens"],
            "sent": sent,
            "queue_s": t["queue_s"],
            "prefill_s": t["prefill_s"],
            "decode_s": t["decode_s"],
            "total_s": t["total_s"],
            "itl_s": t["itl_s"],
            "ttft_s": t["queue_s"] + t["prefill_s"],
        }

    try:
        rows = await asyncio.gather(*(one(i, p) for i, p in enumerate(BENCH_PROMPTS)))
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    wall = time.perf_counter() - t0
    tokens = sum(r["output_tokens"] for r in rows)
    max_batch = meta.get("max_batch") or 1

    # rows held vs rows actually generating, sampled over the run
    grid, occupancy = 120, []
    for j in range(grid):
        t = wall * j / (grid - 1)
        held = sum(1 for r in rows if r["sent"] <= t < r["sent"] + r["total_s"])
        gen = sum(1 for r in rows
                  if r["sent"] + r["ttft_s"] <= t < r["sent"] + r["total_s"])
        occupancy.append({"t": round(t, 3), "generating": gen, "held": held})

    mean_live = sum(o["generating"] for o in occupancy) / len(occupancy)

    return {
        "context": {
            "device": meta.get("device"),
            "max_batch": max_batch,
            "prompts": len(rows),
            "output_tokens": tokens,
            "wall_s": wall,
        },
        "headline": {
            "throughput_tps": tokens / wall,
            "ttft_p99_s": _pct([r["ttft_s"] for r in rows], 99),
            "itl_p50_s": _pct([r["itl_s"] for r in rows], 50),
            "occupancy": mean_live / max_batch,
        },
        "percentiles": {
            "ttft_s": _spread([r["ttft_s"] for r in rows]),
            "itl_s": _spread([r["itl_s"] for r in rows]),
            "latency_s": _spread([r["total_s"] for r in rows]),
        },
        "requests": rows,
        "occupancy": occupancy,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

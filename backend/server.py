"""CPU tier: the public API. Owns everything that is not a forward pass.

    uv run uvicorn backend.server:app --port 8000

Right now it validates a request and forwards it to the GPU tier at MODEL_URL.
Auth, sessions, conversation history, rate limits and billing land here rather
than in model/server.py, so the GPU tier stays swappable -- for vLLM, for
Bedrock, for a second model -- without any of that moving with it.

/api/chat is a pass-through stream: bytes from the GPU tier are forwarded to the
browser as they arrive, so nothing here buffers a whole answer.
"""

import asyncio
import json
import math
import os
import random
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from backend.prompts import PROMPTS

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8080")
ORIGINS = ["http://localhost:3000", "https://socratesllm.vercel.app"]
TIMEOUT_S = 300
BENCH_TIMEOUT_S = 1800
BENCH_RATE = 3.0      # requests/sec
BENCH_SEED = 0

client = httpx.AsyncClient(
    base_url=MODEL_URL, timeout=TIMEOUT_S,
    limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
)


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
async def chat(req: ChatRequest) -> StreamingResponse:
    # auth, rate limit and history load go here, before the prompt is assembled
    started = time.perf_counter()
    upstream = client.build_request("POST", "/stream", json={"prompt": req.prompt})
    try:
        r = await client.send(upstream, stream=True)
    except httpx.RequestError as e:
        EDGE_REQUESTS.labels("chat", "502").inc()
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    if r.status_code != 200:
        # pass the model tier's own 413/429/504 through rather than masking it
        detail = (await r.aread()).decode()
        await r.aclose()
        EDGE_REQUESTS.labels("chat", str(r.status_code)).inc()
        raise HTTPException(status_code=r.status_code,
                            detail=json.loads(detail).get("detail", detail))

    async def relay():
        # a closed tab cancels this, which closes the upstream response, which is
        # what tells the GPU tier to stop generating
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await r.aclose()
            EDGE.observe(time.perf_counter() - started)
            EDGE_REQUESTS.labels("chat", "200").inc()

    return StreamingResponse(relay(), media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no"})


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
    return {f"p{p}": _pct(xs, p) for p in (50, 90, 95)}


class BenchRequest(BaseModel):
    rate: float = BENCH_RATE
    seed: int = BENCH_SEED


@app.post("/api/benchmark")
async def benchmark(req: BenchRequest | None = None) -> dict:
    """Open-loop: arrivals are Poisson at `rate`, not one burst.

    Seeded, so the same rate and seed replay the same arrival sequence. Firing
    everything at t=0 measures how a full batch drains; a rate measures what the
    scheduler sustains, which is the number that generalises.
    """
    req = req or BenchRequest()
    try:
        meta = (await client.get("/health", timeout=10.0)).json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    max_batch = meta.get("max_batch") or 1

    # One throwaway request before the clock starts. The decode graphs for every
    # row count are compiled at model-tier startup (Engine.warmup), which is the
    # only place that can do it reliably -- this is just to confirm the tier
    # answers and to keep any first-request lazy work out of the measurement.
    try:
        await client.post("/generate", json={"prompt": "say hi"},
                          timeout=BENCH_TIMEOUT_S)
    except httpx.RequestError:
        pass                      # a failed warmup is not worth failing the run

    t0 = time.perf_counter()

    async def one(i: int, bucket: str, prompt: str, delay: float) -> dict:
        await asyncio.sleep(delay)
        sent = time.perf_counter() - t0
        r = await client.post("/generate", json={"prompt": prompt}, timeout=BENCH_TIMEOUT_S)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        b = r.json()
        t = b["timing"]
        received = time.perf_counter() - t0
        started = received - t["total_s"]
        return {
            "i": i,
            "started": started,
            "first_token": started + t["queue_s"] + t["prefill_s"],
            "finished": received,
            "bucket": bucket,
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
        rng = random.Random(req.seed)
        at, delays = 0.0, []
        for _ in PROMPTS:
            delays.append(at)
            at += rng.expovariate(req.rate)
        rows = await asyncio.gather(
            *(one(i, bucket, p, d)
              for i, ((bucket, p), d) in enumerate(zip(PROMPTS, delays)))
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    wall = time.perf_counter() - t0
    tokens = sum(r["output_tokens"] for r in rows)

    # Three series, and the distinction matters. A queued request holds no row --
    # it sits in the engine's pending deque -- so counting it as "held" made this
    # exceed MAX_BATCH (all 15 at t=0 against 4 real rows). Held now starts at
    # admission, so held minus generating is prefill, which is genuine occupied
    # capacity not yet producing tokens.
    grid, occupancy = 120, []
    for j in range(grid):
        t = wall * j / (grid - 1)
        queued = sum(1 for r in rows if r["sent"] <= t < r["started"] + r["queue_s"])
        held = sum(1 for r in rows
                   if r["started"] + r["queue_s"] <= t < r["finished"])
        gen = sum(1 for r in rows if r["first_token"] <= t < r["finished"])
        occupancy.append({"t": round(t, 3), "generating": gen, "held": held,
                          "queued": queued})

    mean_live = sum(o["generating"] for o in occupancy) / len(occupancy)
    # ITL is a per-gap measure, so a request that emitted one token has none and
    # would otherwise report its whole decode span as a single interval. Use the
    # median rather than the mean too: one such outlier at 46s is enough to make
    # a mean meaningless.
    itls = [r["itl_s"] for r in rows if r["output_tokens"] >= 2]
    itl_p50 = _pct(itls, 50)

    return {
        "context": {
            "device": meta.get("device"),
            "max_batch": max_batch,
            "prompts": len(rows),
            "rate": req.rate,
            "seed": req.seed,
            "output_tokens": tokens,
            "wall_s": wall,
            "itl_sample": len(itls),
        },
        "headline": {
            "throughput_tps": tokens / wall,
            # Per stream, so aggregate cannot be mistaken for what one caller
            # sees. With N rows sharing a decode step, aggregate is roughly N x.
            "per_stream_tps": 1 / itl_p50 if itl_p50 else 0.0,
            "ttft_p95_s": _pct([r["ttft_s"] for r in rows], 95),
            "itl_p50_s": itl_p50,
            "occupancy": mean_live / max_batch,
        },
        "percentiles": {
            "ttft_s": _spread([r["ttft_s"] for r in rows]),
            "itl_s": _spread(itls),
            "latency_s": _spread([r["total_s"] for r in rows]),
        },
        "requests": rows,
        "occupancy": occupancy,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

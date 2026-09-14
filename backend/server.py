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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8080")
ORIGINS = ["http://localhost:3000"]
TIMEOUT_S = 300
BENCH_TIMEOUT_S = 1800     # 16 prompts through MAX_BATCH rows takes minutes

# Five prompts each of short, medium and long expected output, so the batch has
# real variance in how long rows live. That variance is the whole point: it is
# what separates continuous batching from static, where one long job holds a
# wave open while short ones sit finished. An all-short set (the previous
# version was 11 of 16) drains too fast to fill rows, which understates
# occupancy and throughput both.
#
# The bucket is an *expectation*, not a guarantee -- the model decides when to
# stop. It is reported per request so the table can be read by class.
BENCH_PROMPTS: list[tuple[str, str]] = [
    # short: a handful of tokens
    ("short", "what is 2+2?"),
    ("short", "capital of France?"),
    ("short", "name a color"),
    ("short", "what is 10-7?"),
    ("short", "say hi"),
    # medium: a sentence or a short list
    ("medium", "in two sentences, what is a KV cache?"),
    ("medium", "name three fruits with one fact about each"),
    ("medium", "write a haiku about GPUs"),
    ("medium", "why does water boil at a lower temperature at altitude?"),
    ("medium", "give me three names for a pet cat"),
    # long: multi-paragraph, several hundred tokens
    ("long", "how to make pizza?"),
    ("long", "explain recursion with a worked example"),
    ("long", "explain how a transformer language model generates text"),
    ("long", "describe the rules of chess to a beginner"),
    ("long", "write a short guide to renting your first apartment"),
]

# One pooled client for the process
client = httpx.AsyncClient(base_url=MODEL_URL, timeout=TIMEOUT_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.aclose()


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
    try:
        r = await client.post("/generate", json={"prompt": req.prompt})
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    if r.status_code != 200:
        # pass the model tier's own 413/429/504 through rather than masking it
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise HTTPException(status_code=r.status_code, detail=detail)

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

    async def one(i: int, bucket: str, prompt: str) -> dict:
        sent = time.perf_counter() - t0
        r = await client.post("/generate", json={"prompt": prompt}, timeout=BENCH_TIMEOUT_S)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        b = r.json()
        t = b["timing"]
        return {
            "i": i,
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
        rows = await asyncio.gather(
            *(one(i, bucket, p) for i, (bucket, p) in enumerate(BENCH_PROMPTS))
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
        queued = sum(1 for r in rows if r["sent"] <= t < r["sent"] + r["queue_s"])
        held = sum(1 for r in rows
                   if r["sent"] + r["queue_s"] <= t < r["sent"] + r["total_s"])
        gen = sum(1 for r in rows
                  if r["sent"] + r["ttft_s"] <= t < r["sent"] + r["total_s"])
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
            "output_tokens": tokens,
            "wall_s": wall,
            "itl_sample": len(itls),
        },
        "headline": {
            "throughput_tps": tokens / wall,
            # Per stream, so aggregate cannot be mistaken for what one caller
            # sees. With N rows sharing a decode step, aggregate is roughly N x.
            "per_stream_tps": 1 / itl_p50 if itl_p50 else 0.0,
            "ttft_p99_s": _pct([r["ttft_s"] for r in rows], 99),
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

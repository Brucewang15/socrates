"""GPU tier: owns the model, the KV cache and the decode loop.

    uv run uvicorn model.server:app --port 8080

Nothing user-facing lives here -- no auth, no sessions, no CORS. It trusts
whatever can reach the port, which locally is your laptop and in AWS is a
security group. backend/server.py is the only thing meant to call it, which is
what keeps this tier a stateless text-in/text-out service.

One background thread owns the device. Request threads only submit and block on
their request's event, which is what lets several in-flight requests share a
decode batch.
"""

import threading
import time

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

import model.inference_cont as cont

IDLE_S = 0.005
TIMEOUT_S = 300
MAX_QUEUE = 16 * cont.MAX_BATCH

# Histograms, not summaries: quantiles have to be computable across replicas,
# and a mean TTFT hides the bimodal shape queueing creates. Buckets are sized
# from measured runs -- TTFT lands between 0.2s and 15s at MAX_BATCH 4.
TTFT = Histogram("socrates_ttft_seconds", "queue wait plus prefill",
                 buckets=(.05, .1, .25, .5, 1, 2, 5, 10, 20, 60))
QUEUE = Histogram("socrates_queue_seconds", "waiting for a row",
                  buckets=(.005, .05, .25, 1, 2, 5, 10, 30))
PREFILL = Histogram("socrates_prefill_seconds", "prompt forward pass",
                    buckets=(.01, .05, .1, .25, .5, 1, 2, 5))
ITL = Histogram("socrates_itl_seconds", "seconds per generated token",
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
LATENCY = Histogram("socrates_request_seconds", "submit to last token",
                    buckets=(.5, 1, 2, 5, 10, 30, 60, 120, 300))

REQUESTS = Counter("socrates_requests_total", "generate calls", ["outcome"])
OUT_TOKENS = Counter("socrates_output_tokens_total", "tokens generated")
IN_TOKENS = Counter("socrates_prompt_tokens_total", "tokens prefilled")

app = FastAPI(title="socrates-model")

# Blocking so we don't start serving when model weights hasn't loaded
engine = cont.Engine()


def serve() -> None:
    """Never returns. Engine.run()'s body, with an idle wait instead of an exit."""
    while True:
        if not (engine.pending or engine.n_active):
            time.sleep(IDLE_S)
            continue
        engine.admit()
        # a request whose first token is a stop token finishes during prefill
        for req in [r for r in engine.rows[:engine.n_active] if r and r.done]:
            engine.retire(req)
        if engine.n_active:
            engine.decode_step()
            for req in [r for r in engine.rows[:engine.n_active] if r and r.done]:
                engine.retire(req)


threading.Thread(target=serve, daemon=True).start()


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    # shed load at the door rather than letting the deque grow without bound
    if len(engine.pending) >= MAX_QUEUE:
        REQUESTS.labels("queue_full").inc()
        raise HTTPException(status_code=429, detail="queue full")
    try:
        r = engine.submit(req.prompt)
    except ValueError as e:
        REQUESTS.labels("too_long").inc()
        raise HTTPException(status_code=413, detail=str(e)) from e

    # the engine sets this in retire(); nothing here holds a reference to r
    # afterwards, so a timed-out request is freed once the engine drops its row
    if not r.event.wait(timeout=TIMEOUT_S):
        REQUESTS.labels("timeout").inc()
        raise HTTPException(status_code=504, detail="generation timed out")
    n = len(r.output)
    queue_s, prefill_s = r.admitted - r.submitted, r.first_token - r.admitted
    decode_s, itl_s = r.finished - r.first_token, (r.finished - r.first_token) / max(n, 1)
    QUEUE.observe(queue_s)
    PREFILL.observe(prefill_s)
    TTFT.observe(queue_s + prefill_s)
    ITL.observe(itl_s)
    LATENCY.observe(r.finished - r.submitted)
    REQUESTS.labels("ok").inc()
    OUT_TOKENS.inc(n)
    IN_TOKENS.inc(int(r.ids.shape[1]))
    return {
        "response": engine.tok.decode(r.output),
        "prompt_tokens": int(r.ids.shape[1]),
        "output_tokens": n,
        "timing": {
            "queue_s": queue_s,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "total_s": r.finished - r.submitted,
            "itl_s": itl_s,
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": cont.DEVICE, "max_batch": cont.MAX_BATCH}


# Sampled at scrape time rather than pushed, so they are always current.
# Queue depth is the number to autoscale on -- GPU utilisation sits near 100%
# during decode whether one row is busy or all of them.
Gauge("socrates_queue_depth", "requests waiting for a row").set_function(
    lambda: len(engine.pending))
Gauge("socrates_active_rows", "rows currently generating").set_function(
    lambda: engine.n_active)
Gauge("socrates_max_batch", "row capacity").set_function(lambda: cont.MAX_BATCH)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics.json")
def metrics_json() -> dict:
    return {
        "pending": len(engine.pending),
        "active": engine.n_active,
        "max_batch": cont.MAX_BATCH,
    }

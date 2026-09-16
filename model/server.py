"""GPU tier: owns the model, the KV cache and the decode loop.

    uv run uvicorn model.server:app --port 8080

Nothing user-facing lives here -- no auth, no sessions, no CORS. It trusts
whatever can reach the port, which locally is your laptop and in AWS is a
security group. backend/server.py is the only thing meant to call it, which is
what keeps this tier a stateless text-in/text-out service.

One background thread owns the device. Handlers are async and never block on it:
the decode thread hands each token to the event loop, so the loop stays free to
answer /metrics and /health while generation is in flight.
"""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

import model.inference_cont as cont
from model.inference_cont import Request

IDLE_S = 0.005
TIMEOUT_S = 300

# Histograms, not summaries: quantiles have to be computable across replicas,
# and a mean TTFT hides the bimodal shape queueing creates. Buckets are sized
# from measured runs. The top bucket has to exceed the worst real TTFT:
# 48 prompts through MAX_BATCH rows is 3 waves, and a long answer is several
# hundred tokens, so the last arrivals can wait minutes. Anything above the
# highest finite bucket lands in +Inf, where histogram_quantile cannot
# interpolate and the percentile silently pins to the top bucket.
TTFT = Histogram("socrates_ttft_seconds", "queue wait plus prefill",
                 buckets=(.05, .1, .25, .5, 1, 2, 5, 10, 20, 60, 120, 300))
QUEUE = Histogram("socrates_queue_seconds", "waiting for a row",
                  buckets=(.005, .05, .25, 1, 2, 5, 10, 30, 60, 120, 300))
PREFILL = Histogram("socrates_prefill_seconds", "prompt forward pass",
                    buckets=(.01, .05, .1, .25, .5, 1, 2, 5))
ITL = Histogram("socrates_itl_seconds", "seconds per generated token",
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
LATENCY = Histogram("socrates_request_seconds", "submit to last token",
                    buckets=(.5, 1, 2, 5, 10, 30, 60, 120, 300, 600))

REQUESTS = Counter("socrates_requests_total", "generate calls", ["outcome"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # the decode thread needs this handle to hand tokens back to the loop
    engine.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="socrates-model", lifespan=lifespan)

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
        for req in engine.rows[:engine.n_active]:
            if req.done or req.cancelled:
                engine.retire(req)
        if engine.n_active:
            engine.decode_step()
            for req in engine.rows[:engine.n_active]:
                if req.done or req.cancelled:
                    engine.retire(req)


threading.Thread(target=serve, daemon=True).start()


class GenerateRequest(BaseModel):
    prompt: str


def submit(prompt: str) -> Request:
    """The engine decides what it can take; this only maps that to a status."""
    try:
        return engine.submit(prompt, stream=True)
    except cont.QueueFull as e:
        REQUESTS.labels("queue_full").inc()
        raise HTTPException(status_code=429, detail=str(e)) from None
    except cont.TooLong as e:
        REQUESTS.labels("too_long").inc()
        raise HTTPException(status_code=413, detail=str(e)) from None


def timing(r: Request) -> dict:
    n = len(r.output)
    queue_s = r.admitted - r.submitted
    prefill_s = r.first_token - r.admitted
    decode_s = r.finished - r.first_token
    itl_s = decode_s / max(n, 1)
    QUEUE.observe(queue_s)
    PREFILL.observe(prefill_s)
    TTFT.observe(queue_s + prefill_s)
    ITL.observe(itl_s)
    LATENCY.observe(r.finished - r.submitted)
    REQUESTS.labels("ok").inc()
    return {"queue_s": queue_s, "prefill_s": prefill_s, "decode_s": decode_s,
            "total_s": r.finished - r.submitted, "itl_s": itl_s}


async def deltas(r: Request):
    """Text as it is generated.

    A token can be a fragment of a character -- an emoji is four byte-level
    tokens -- and the tokenizer renders an incomplete tail as U+FFFD until the
    next token completes it. Sending that tail would put a replacement char on
    the wire and then never send the real one, since the correction is not an
    append. So decode the whole prefix each time and hold back the unstable end.
    """
    sent = ""
    while await asyncio.wait_for(r.stream.get(), timeout=TIMEOUT_S) is not None:
        text = engine.tok.decode(r.output).rstrip("\ufffd")
        if len(text) > len(sent):
            yield text[len(sent):]
            sent = text
    # the model can stop mid-character; nothing is coming to complete it
    text = engine.tok.decode(r.output)
    if len(text) > len(sent):
        yield text[len(sent):]


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    """Buffered, same shape as before. Async only so that waiting for the last
    token costs an idle coroutine instead of a threadpool thread."""
    r = submit(req.prompt)
    try:
        async for _ in deltas(r):
            pass
    except TimeoutError:
        r.cancelled = True
        REQUESTS.labels("timeout").inc()
        raise HTTPException(status_code=504, detail="generation timed out") from None
    return {
        "response": engine.tok.decode(r.output),
        "prompt_tokens": int(r.ids.shape[1]),
        "output_tokens": len(r.output),
        "timing": timing(r),
    }


@app.post("/stream")
async def stream(req: GenerateRequest) -> StreamingResponse:
    """One JSON object per line: {"delta": ...} per token, {"done": true, ...} last."""
    r = submit(req.prompt)

    async def lines():
        try:
            async for delta in deltas(r):
                yield json.dumps({"delta": delta}) + "\n"
        except (asyncio.CancelledError, TimeoutError):
            # closed tab: stop paying for tokens nobody will read
            r.cancelled = True
            raise
        yield json.dumps({"done": True, "prompt_tokens": int(r.ids.shape[1]),
                          "output_tokens": len(r.output),
                          "timing": timing(r)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"X-Accel-Buffering": "no"})


@app.get("/health")
async def health() -> dict:
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
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics.json")
async def metrics_json() -> dict:
    return {
        "pending": len(engine.pending),
        "active": engine.n_active,
        "max_batch": cont.MAX_BATCH,
    }

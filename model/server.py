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

import model.inference_cont as cont
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

IDLE_S = 0.005
TIMEOUT_S = 300
MAX_QUEUE = 16 * cont.MAX_BATCH

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
        raise HTTPException(status_code=429, detail="queue full")
    try:
        r = engine.submit(req.prompt)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    # the engine sets this in retire(); nothing here holds a reference to r
    # afterwards, so a timed-out request is freed once the engine drops its row
    if not r.event.wait(timeout=TIMEOUT_S):
        raise HTTPException(status_code=504, detail="generation timed out")
    n = len(r.output)
    return {
        "response": engine.tok.decode(r.output),
        "prompt_tokens": int(r.ids.shape[1]),
        "output_tokens": n,
        "timing": {
            "queue_s": r.admitted - r.submitted,
            "prefill_s": r.first_token - r.admitted,
            "decode_s": r.finished - r.first_token,
            "total_s": r.finished - r.submitted,
            "itl_s": (r.finished - r.first_token) / max(n, 1),
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": cont.DEVICE, "max_batch": cont.MAX_BATCH}


@app.get("/metrics")
def metrics() -> dict:
    """Queue depth is the number to autoscale on -- GPU utilisation sits near
    100% during decode whether one row is busy or all of them."""
    return {
        "pending": len(engine.pending),
        "active": engine.n_active,
        "max_batch": cont.MAX_BATCH,
    }

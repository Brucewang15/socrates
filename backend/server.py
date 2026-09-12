"""HTTP server for the socrates inference engine.

    uv run uvicorn backend.server:app --port 8000

Engine.run() drains a queue and returns; a server needs the same steps on a
loop that idles instead of exiting, so that loop lives here. One background
thread owns the model, the cache and the rows. Request threads only call
engine.submit() and block on the request's event, which is what lets several
in-flight requests share a decode batch.
"""

import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference_cont import MAX_BATCH, Engine

IDLE_S = 0.005      # no live rows and nothing queued; wait before looking again
TIMEOUT_S = 300
MAX_QUEUE = 16 * MAX_BATCH   # bound the backlog; deque growth is the one real leak

app = FastAPI(title="socrates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()


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


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    try:
        r = engine.submit(req.message)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    # the engine sets this in retire(); nothing here holds a reference to r
    # afterwards, so a timed-out request is freed once the engine drops its row
    if not r.event.wait(timeout=TIMEOUT_S):
        raise HTTPException(status_code=504, detail="generation timed out")
    return {"response": engine.tok.decode(r.output)}

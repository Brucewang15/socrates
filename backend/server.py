"""CPU tier: the public API. Owns everything that is not a forward pass.

    uv run uvicorn backend.server:app --port 8000

Right now it validates a request and forwards it to the GPU tier at MODEL_URL.
Auth, sessions, conversation history, rate limits and billing land here rather
than in model/server.py, so the GPU tier stays swappable -- for vLLM, for
Bedrock, for a second model -- without any of that moving with it.

No streaming yet: this holds the request open and returns the whole string.
"""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Same reason DEVICE is an env var: a container cannot reach localhost:8080.
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8080")
ORIGINS = ["http://localhost:3000"]
TIMEOUT_S = 300

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
async def chat(req: ChatRequest) -> dict[str, str]:
    try:
        r = await client.post("/generate", json={"prompt": req.prompt})
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"model tier unreachable at {MODEL_URL}") from e

    if r.status_code != 200:
        # pass the model tier's own 413/429/504 through rather than masking it
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise HTTPException(status_code=r.status_code, detail=detail)

    return {"response": r.json()["response"]}


@app.get("/health")
async def health() -> dict:
    """Reports the downstream too: this tier is useless without it."""
    try:
        r = await client.get("/health", timeout=2.0)
        return {"status": "ok", "model": r.json()}
    except httpx.RequestError:
        return {"status": "degraded", "model": None}

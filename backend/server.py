"""
HTTP server for the socrates inference engine.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="socrates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    # TODO: hand off to the inference engine once batching and the scheduler exist.
    return {"response": "Work in progress — the inference engine isn't wired up yet."}

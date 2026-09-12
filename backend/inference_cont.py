"""Continuous batching: a finished row is refilled from the queue the same step.

    uv run -m backend.inference_cont

Prefill is sequential -- one request at a time, its own length, no padding.
Decode runs every live row together.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import torch
from model.qwen_kv_cont import MODEL_ID, KVCache, Qwen3, load_config, load_weights
from transformers import AutoTokenizer

MAX_BATCH = 4
MAX_NEW_TOKENS = 1024
MAX_LEN = 2048
DEVICE = "mps"
DTYPE = torch.bfloat16

PROMPTS = [
    "how to make pizza?",
    "what is 2+2?",
    "who are you?",
    "name a color",
    "capital of France?",
    "name a fruit",
    "explain recursion",
    "say hi",
    "what is 5*5?",
    "name an animal",
    "how does a KV cache work?",
    "what is 10-7?",
    "name a country",
    "write a haiku about GPUs",
    "what is the boiling point of water?",
    "name a programming language",
]


@dataclass
class Request:
    prompt: str
    ids: torch.Tensor | None = None
    output: list[int] = field(default_factory=list)
    done: bool = False
    row: int = -1
    submitted: float = field(default_factory=time.perf_counter)
    first_token: float = 0.0
    finished: float = 0.0


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.cfg = load_config()
        with torch.device("meta"):
            model = Qwen3(self.cfg)
        model.load_state_dict(load_weights(), assign=True)
        self.model = model.eval().to(DEVICE)
        self.cache = KVCache(MAX_BATCH, self.cfg["num_hidden_layers"],
                             self.cfg["num_key_value_heads"], self.cfg["head_dim"],
                             max_len=MAX_LEN, dtype=DTYPE, device=DEVICE)
        self.rows: list[Request | None] = [None] * MAX_BATCH
        self.n_active = 0
        self.pending: deque[Request] = deque()

    def submit(self, prompt: str) -> Request:
        req = Request(prompt)
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        req.ids = self.tok(text, return_tensors="pt").input_ids.to(DEVICE)
        self.pending.append(req)
        return req

    def prefill(self, req: Request, row: int) -> None:
        rows = slice(row, row + 1)
        positions = torch.arange(req.ids.shape[1], device=DEVICE)[None]
        logits = self.model(req.ids, self.cache, rows, positions)
        self.record(req, int(logits[:, -1].argmax(-1)))

    def admit(self) -> None:
        while self.pending and self.n_active < MAX_BATCH:
            req = self.pending.popleft()
            row = self.n_active
            self.cache.reset(row)
            self.rows[row] = req
            req.row = row
            self.n_active += 1
            self.prefill(req, row)

    def decode_step(self) -> None:
        live = self.rows[:self.n_active]
        ids = torch.tensor([[r.output[-1]] for r in live], device=DEVICE)
        positions = torch.tensor([[self.cache.lengths[r.row]] for r in live], device=DEVICE)
        logits = self.model(ids, self.cache, slice(0, self.n_active), positions)
        for req, token in zip(live, logits[:, -1].argmax(-1).tolist()):
            self.record(req, int(token))

    def record(self, req: Request, token: int) -> None:
        now = time.perf_counter()
        if not req.first_token:
            req.first_token = now
        if token in self.tok.all_special_ids or len(req.output) >= MAX_NEW_TOKENS:
            req.done = True
            req.finished = now
        else:
            req.output.append(token)

    def retire(self, req: Request) -> None:
        row, last = req.row, self.n_active - 1
        if row != last:
            self.cache.move_row(last, row)
            self.rows[row] = self.rows[last]
            self.rows[row].row = row
        self.rows[last] = None
        self.n_active -= 1

    def run(self) -> None:
        while self.pending or self.n_active:
            self.admit()
            for req in [r for r in self.rows[:self.n_active] if r and r.done]:
                self.retire(req)
            if self.n_active:
                self.decode_step()
            for req in [r for r in self.rows[:self.n_active] if r and r.done]:
                self.retire(req)


def main() -> None:
    engine = Engine()
    reqs = [engine.submit(p) for p in PROMPTS]
    print(f"{len(reqs)} requests, {MAX_BATCH} rows\n")

    t0 = time.perf_counter()
    engine.run()
    wall = time.perf_counter() - t0

    for r in reqs:
        print(f"> {r.prompt}\n{engine.tok.decode(r.output)}\n")
    total = sum(len(r.output) for r in reqs)
    ttft = sorted((r.first_token - r.submitted) for r in reqs)
    print(f"{wall:.2f}s | {total} tok | {total / wall:.2f} tok/s | "
          f"ttft min {ttft[0]:.2f}s max {ttft[-1]:.2f}s")


if __name__ == "__main__":
    main()

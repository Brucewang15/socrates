"""Continuous batching: a finished row is refilled from the queue the same step.

    uv run -m model.inference_cont

Prefill is sequential -- one request at a time, its own length, no padding.
Decode runs every live row together.
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from model.qwen.qwen_kv_cont import MODEL_ID, KVCache, Qwen3, load_config, load_weights
from transformers import AutoTokenizer

MAX_BATCH = 24
MAX_NEW_TOKENS = 1024
MAX_LEN = 2048
DEVICE = os.getenv("DEVICE", "mps")
DTYPE = torch.bfloat16
# COMPILE=0 to fall back to eager decode -- worth having when torch.compile
# graph-breaks on the cache bookkeeping, or when A/B-ing the speedup.
COMPILE = os.getenv("COMPILE", "1") not in ("0", "false", "False")

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
    # set when the request is given a row: submitted->admitted is queue wait,
    # which the scheduler owns; admitted->first_token is prefill, which it does not
    admitted: float = 0.0
    first_token: float = 0.0
    finished: float = 0.0
    # set when the request retires, so a serving thread can block on one
    # request instead of polling. The demo path ignores it.
    event: threading.Event = field(default_factory=threading.Event)


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.cfg = load_config()
        with torch.device("meta"):
            model = Qwen3(self.cfg)
        model.load_state_dict(load_weights(), assign=True)
        self.model = model.eval().to(DEVICE)

        # Prefill stays eager and decode gets compiled, on purpose.
        #
        # Decode is the same shape every step (T=1), runs thousands of times, and
        # was measured at 105 ms wall for 15 ms of GPU work -- ~2,300 kernel
        # launches per step, with the card idle 85% of the time. reduce-overhead
        # wraps the graph in CUDA graphs, which is what collapses those launches.
        #
        # Prefill is a different length for every prompt, so compiling it would
        # recompile per length for no gain: one big forward already amortises
        # launch cost over the whole prompt.
        #
        # Only on CUDA. torch.compile on MPS is far less mature, and
        # reduce-overhead means nothing without CUDA graphs.
        self.eager_model = self.model
        self.decode_model = self.model
        if COMPILE and DEVICE.startswith("cuda"):
            self.decode_model = torch.compile(
                self.model, mode="reduce-overhead", dynamic=False
            )
            print("decode path compiled (mode=reduce-overhead); "
                  "first steps pay compilation and graph capture", flush=True)

        self.cache = KVCache(MAX_BATCH, self.cfg["num_hidden_layers"],
                             self.cfg["num_key_value_heads"], self.cfg["head_dim"],
                             max_len=MAX_LEN, dtype=DTYPE, device=DEVICE)
        self.rows: list[Request | None] = [None] * MAX_BATCH
        self.n_active = 0
        self.pending: deque[Request] = deque()
        self.warmup()

    @torch.no_grad()
    def warmup(self) -> None:
        """Compile the decode graph for every batch width before serving.

        dynamic=False means one trace per distinct row count: the graph built
        for 4 rows is not reused at 3. Letting real traffic discover those
        widths costs ~45s the first time the batch drains 4->3, and again
        3->2 -- paid mid-request, charged to whichever rows are resident. A
        measured run showed exactly that: throughput fell from 126 tok/s to
        near zero for 48s at each transition, and the stall landed inside the
        reported inter-token latency of three unrelated requests.

        Doing it here rather than by sending warmup requests through the
        scheduler, because that cannot reach a given width on purpose --
        concurrent requests arrive milliseconds apart, and identical prompts
        produce identical output and retire on the same step, so a wave of 4
        drains 4->1 and never touches 3 or 2.
        """
        if self.decode_model is self.model:
            return                       # eager decode: nothing to trace
        t0 = time.perf_counter()
        ids = torch.zeros((MAX_BATCH, 1), dtype=torch.long, device=DEVICE)
        for width in range(1, MAX_BATCH + 1):
            # more than one step per width: reduce-overhead defers CUDA graph
            # capture past the first call, so a single step can leave the
            # capture itself for the real run to pay.
            for step in range(3):
                positions = torch.full((width, 1), step, dtype=torch.long, device=DEVICE)
                self.decode_model(ids[:width], self.cache, slice(0, width), positions)
        # warmup wrote real tokens into rows 0..MAX_BATCH-1; hand them back empty
        for row in range(MAX_BATCH):
            self.cache.reset(row)
        print(f"decode graphs warm for 1..{MAX_BATCH} rows "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)

    def submit(self, prompt: str) -> Request:
        req = Request(prompt)
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        req.ids = self.tok(text, return_tensors="pt").input_ids
        if req.ids.shape[1] + MAX_NEW_TOKENS > MAX_LEN:
            raise ValueError(f"prompt is {req.ids.shape[1]} tokens; with "
                             f"{MAX_NEW_TOKENS} new it exceeds the {MAX_LEN}-token row")
        self.pending.append(req)
        return req

    @torch.no_grad()
    def prefill(self, req: Request, row: int) -> None:
        rows = slice(row, row + 1)
        # the only thread that touches the device is the one running the loop
        ids = req.ids.to(DEVICE)
        positions = torch.arange(ids.shape[1], device=DEVICE)[None]
        logits = self.eager_model(ids, self.cache, rows, positions)
        # the cache no longer tracks this for us -- see KVCache.append
        self.cache.lengths[row] = ids.shape[1]
        self.record(req, int(logits[:, -1].argmax(-1)))

    def admit(self) -> None:
        while self.pending and self.n_active < MAX_BATCH:
            req = self.pending.popleft()
            row = self.n_active
            self.cache.reset(row)
            self.rows[row] = req
            req.row = row
            req.admitted = time.perf_counter()
            self.n_active += 1
            self.prefill(req, row)

    @torch.no_grad()
    def decode_step(self) -> None:
        live = self.rows[:self.n_active]
        ids = torch.tensor([[r.output[-1]] for r in live], device=DEVICE)
        positions = torch.tensor([[self.cache.lengths[r.row]] for r in live], device=DEVICE)
        logits = self.decode_model(ids, self.cache, slice(0, self.n_active), positions)
        # every live row wrote exactly one slot; the cache leaves this to us
        for req in live:
            self.cache.lengths[req.row] += 1
        # Do not hold on to `logits` past this point: under CUDA graphs the
        # output buffer is reused by the next replay. tolist() copies to host
        # here and now, which is safe; stashing the tensor would not be.
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
        req.event.set()

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

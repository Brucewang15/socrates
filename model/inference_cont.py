"""Continuous batching: a finished row is refilled from the queue the same step.

    uv run -m model.inference_cont

Prefill is sequential -- one request at a time, its own length, no padding.
Decode runs every live row together.
"""

import asyncio
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from prometheus_client import Counter
from transformers import AutoTokenizer

from model.qwen.qwen_kv_cont import MODEL_ID, KVCache, Qwen3, load_config, load_weights

MAX_BATCH = 18
MAX_NEW_TOKENS = 1024
MAX_LEN = 2048
MAX_QUEUE = 16 * MAX_BATCH
# How coarsely the KV read window is rounded up. Smaller reads fewer bytes per
# step; larger means fewer distinct shapes, so fewer graphs to compile and warm.
# The shape has to be constant, not maximal -- see KVCache.append.
WINDOW_BUCKET = int(os.getenv("WINDOW_BUCKET", "512"))
DEVICE = os.getenv("DEVICE", "mps")
DTYPE = torch.bfloat16

# counted as tokens are produced, not once the request returns, so a scrape
# during a long generation sees the work in progress
OUT_TOKENS = Counter("socrates_output_tokens_total", "tokens generated")
IN_TOKENS = Counter("socrates_prompt_tokens_total", "tokens prefilled")


class QueueFull(Exception):
    pass


class TooLong(Exception):
    pass
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
    # set by a handler whose client went away; the engine drops the row
    cancelled: bool = False
    event: threading.Event = field(default_factory=threading.Event)
    # tokens leave here one at a time, None last. Only set for streaming calls.
    stream: asyncio.Queue | None = None


class Engine:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.cfg = load_config()
        with torch.device("meta"):
            model = Qwen3(self.cfg)
        model.load_state_dict(load_weights(), assign=True)
        self.model = model.eval().to(DEVICE)

        # Built here rather than in Qwen3.__init__ because the model above is
        # constructed on the meta device, and a meta buffer cannot be copied to a
        # real device. Attached, not passed to forward: that is what keeps its 72
        # tensors module state instead of graph inputs, which is what lets
        # reduce-overhead capture CUDA graphs. See KVCache.
        self.cache = KVCache(MAX_BATCH, self.cfg["num_hidden_layers"],
                             self.cfg["num_key_value_heads"], self.cfg["head_dim"],
                             max_len=MAX_LEN, dtype=DTYPE, device=DEVICE)
        self.model.attach_cache(self.cache)

        # The two tensors a decode step feeds in, allocated once. Freshly built
        # per step they were a host->device allocation on the critical path of the
        # very loop whose overhead we are cutting; now each step is a copy into
        # storage that never moves, which is also what CUDA graphs want to see.
        self.step_ids = torch.zeros((MAX_BATCH, 1), dtype=torch.long, device=DEVICE)
        self.step_pos = torch.zeros((MAX_BATCH, 1), dtype=torch.long, device=DEVICE)

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

        self.loop: asyncio.AbstractEventLoop | None = None
        self.rows: list[Request | None] = [None] * MAX_BATCH
        self.n_active = 0
        self.pending: deque[Request] = deque()
        self.warmup()

    def window_end(self, longest: int) -> int:
        """Slots to read for a step whose longest live row is at `longest`.

        Rounded up to WINDOW_BUCKET so the shape repeats: the graph captured for a
        512-slot window is reused for every step until some row passes 512.
        """
        need = longest + 1                      # the slot being written now
        buckets = -(-need // WINDOW_BUCKET) * WINDOW_BUCKET
        return min(MAX_LEN, max(WINDOW_BUCKET, buckets))

    def buckets(self) -> list[int]:
        """Every window a decode step can ask for, so warmup can cover them all."""
        return list(range(WINDOW_BUCKET, MAX_LEN + 1, WINDOW_BUCKET)) or [MAX_LEN]

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
        self.step_ids.zero_()
        # One graph per (row count, window) pair now, because both are guarded.
        # That is MAX_BATCH x len(buckets) traces, and every one of them has to be
        # paid here: a bucket first reached mid-run costs its compile inside a
        # live request, which is what this whole method exists to prevent.
        for end in self.buckets():
            for width in range(1, MAX_BATCH + 1):
                # more than one step per width: reduce-overhead defers CUDA graph
                # capture past the first call, so a single step can leave the
                # capture itself for the real run to pay.
                for step in range(3):
                    self.step_pos[:width].fill_(step)
                    self.decode_model(self.step_ids[:width], slice(0, width),
                                      self.step_pos[:width], end)
        # warmup wrote real tokens into rows 0..MAX_BATCH-1; hand them back empty
        for row in range(MAX_BATCH):
            self.cache.reset(row)
        print(f"decode graphs warm for 1..{MAX_BATCH} rows x "
              f"{len(self.buckets())} windows {self.buckets()} "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)

    def submit(self, prompt: str, stream: bool = False) -> Request:
        # shed load at the door rather than letting the deque grow without bound
        if len(self.pending) >= MAX_QUEUE:
            raise QueueFull(f"{len(self.pending)} requests already waiting")
        req = Request(prompt)
        if stream:
            req.stream = asyncio.Queue()
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        req.ids = self.tok(text, return_tensors="pt").input_ids
        if req.ids.shape[1] + MAX_NEW_TOKENS > MAX_LEN:
            raise TooLong(f"prompt is {req.ids.shape[1]} tokens; with "
                          f"{MAX_NEW_TOKENS} new it exceeds the {MAX_LEN}-token row")
        self.pending.append(req)
        return req

    @torch.no_grad()
    def prefill(self, req: Request, row: int) -> None:
        rows = slice(row, row + 1)
        # the only thread that touches the device is the one running the loop
        ids = req.ids.to(DEVICE)
        positions = torch.arange(ids.shape[1], device=DEVICE)[None]
        # eager, so no shape guard to satisfy: read exactly the prompt
        logits = self.eager_model(ids, rows, positions, ids.shape[1])
        # the cache no longer tracks this for us -- see KVCache.append
        self.cache.lengths[row] = ids.shape[1]
        IN_TOKENS.inc(ids.shape[1])
        self.record(req, int(logits[:, -1].argmax(-1)))

    def admit(self) -> None:
        while self.pending and self.n_active < MAX_BATCH:
            req = self.pending.popleft()
            if req.cancelled:
                continue
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
        n = self.n_active
        # Copy into the preallocated inputs rather than allocating two device
        # tensors per step. Built on the host first so this is one H2D copy each
        # into storage whose address never changes.
        self.step_ids[:n].copy_(
            torch.tensor([[r.output[-1]] for r in live], dtype=torch.long))
        pos = [self.cache.lengths[r.row] for r in live]
        self.step_pos[:n].copy_(torch.tensor(pos, dtype=torch.long)[:, None])
        # No sync to find this: lengths are ours, on the host, already.
        end = self.window_end(max(pos))
        logits = self.decode_model(self.step_ids[:n], slice(0, n),
                                   self.step_pos[:n], end)
        # every live row wrote exactly one slot; the cache leaves this to us
        for req in live:
            self.cache.lengths[req.row] += 1
        # Do not hold on to `logits` past this point: under CUDA graphs the
        # output buffer is reused by the next replay. tolist() copies to host
        # here and now, which is safe; stashing the tensor would not be.
        for req, token in zip(live, logits[:, -1].argmax(-1).tolist()):
            self.record(req, int(token))

    def emit(self, req: Request, item: int | None) -> None:
        """Hand a token to the handler waiting on this request. asyncio.Queue is
        not thread-safe and this runs on the decode thread, so it goes through
        the loop rather than straight into the queue."""
        if req.stream is not None:
            self.loop.call_soon_threadsafe(req.stream.put_nowait, item)

    def record(self, req: Request, token: int) -> None:
        now = time.perf_counter()
        if not req.first_token:
            req.first_token = now
        if token in self.tok.all_special_ids or len(req.output) >= MAX_NEW_TOKENS:
            req.done = True
            req.finished = now
            self.emit(req, None)
        else:
            req.output.append(token)
            OUT_TOKENS.inc()
            self.emit(req, token)

    def retire(self, req: Request) -> None:
        row, last = req.row, self.n_active - 1
        if row != last:
            self.cache.move_row(last, row)
            self.rows[row] = self.rows[last]
            self.rows[row].row = row
        self.rows[last] = None
        self.n_active -= 1
        if not req.done:
            req.finished = time.perf_counter()
            self.emit(req, None)
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

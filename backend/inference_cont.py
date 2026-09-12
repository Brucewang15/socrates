"""Continuous batching: a finished row is refilled from the queue the same step.

    uv run -m backend.inference_cont              # demo on the real model
    uv run -m backend.inference_cont --selftest   # equivalence check, tiny random weights

The split mirrors the static pair:

    model/qwen_kv_cont.py        ragged cache + forward pass   (no scheduling)
    backend/inference_cont.py    the scheduler                 (this file)

    model/qwen_kv_seq.py         padded cache + forward pass
    backend/inference_static.py  the wave loop

Prefill is sequential -- one request at a time, at its own length, no padding --
and then merges into the decode batch. Decode runs every live row together, each
row at its own absolute position.

Admission is the entire policy, and it is one line in admit(): fill every free
row, every step. Nobody waits for a wave to drain. Eviction is O(one row)
because model/qwen_kv_cont.BatchedKVCache keeps the live rows packed, and
that is what makes refilling a seat mid-flight cheap enough to bother.
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from model.qwen_kv_cont import (
    MODEL_ID,
    TINY,
    BatchedKVCache,
    Qwen3Continuous,
    load_config,
    load_weights,
    pick_device,
    sync,
)
from transformers import AutoTokenizer

MAX_BATCH = 4
MAX_NEW_TOKENS = 1024
MAX_LEN = 2048
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
    """One inference request and everything needed to score it afterwards."""

    rid: int
    input_ids: torch.Tensor          # [1, P]
    max_new_tokens: int
    prompt: str = ""
    label: str = ""
    arrival: float = 0.0
    row: int = -1
    out_ids: list[int] = field(default_factory=list)
    t_admit: float | None = None
    t_first_token: float | None = None
    t_done: float | None = None
    finish: str = ""                 # "eos" or "length"

    @property
    def queued(self) -> float:
        """How long the scheduler made it wait. The static-vs-continuous gap."""
        return self.t_admit - self.arrival

    @property
    def prompt_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_generated(self) -> int:
        return len(self.out_ids)

    @property
    def ttft(self) -> float:
        """Time to first token, from arrival."""
        return self.t_first_token - self.arrival

    @property
    def latency(self) -> float:
        """End-to-end, from arrival to last token."""
        return self.t_done - self.arrival


@dataclass
class RunResult:
    """Timing for one run. Shared with the static bench so metrics line up."""

    policy: str
    requests: list[Request]
    wall: float
    step_t: list[float]              # seconds since start, one per forward
    step_batch: list[int]            # rows doing useful work in that forward
    step_ms: list[float]
    step_kind: list[str]             # "prefill" or "decode"
    cache_bytes: int

    @property
    def out_tokens(self) -> int:
        return sum(r.n_generated for r in self.requests)

    @property
    def throughput(self) -> float:
        return self.out_tokens / self.wall

    @property
    def mean_batch(self) -> float:
        decode = [b for b, k in zip(self.step_batch, self.step_kind) if k == "decode"]
        return sum(decode) / len(decode) if decode else 0.0


class Engine:
    """Continuous-batching scheduler over the ragged cache.

    Prefill runs on its own for each admitted request, then merges into the
    decode batch. Interleaving prefill with decode in one forward is the better
    design and is left for later; this one keeps the admission logic -- the
    thing being measured -- readable.
    """

    def __init__(self, model: Qwen3Continuous, cfg: dict, *, stop_ids, max_batch: int,
                 max_len: int):
        p = next(model.parameters())
        self.model = model
        self.cfg = cfg
        self.device = p.device
        self.policy = "continuous"
        self.max_batch = max_batch
        self.max_len = max_len
        self.stop_ids = set(stop_ids)
        self.cache = BatchedKVCache(
            cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"],
            max_batch=max_batch, max_len=max_len, dtype=p.dtype, device=p.device,
        )
        self.waiting: deque[Request] = deque()
        self.rows: list[Request | None] = [None] * max_batch
        self.n_active = 0

    # ---- admission control -------------------------------------------------

    def _free_rows(self) -> int:
        return self.max_batch - self.n_active

    def _admissible(self, req: Request) -> bool:
        """Would this request fit? The day-1 capacity arithmetic, as code."""
        return req.prompt_len + req.max_new_tokens <= self.max_len

    def admit(self, now: float) -> list[Request]:
        """Fill every free row, every step. That is the whole policy."""
        admitted = []
        while self.waiting and self._free_rows() > 0:
            req = self.waiting[0]
            if not self._admissible(req):
                raise ValueError(
                    f"request {req.rid} needs {req.prompt_len + req.max_new_tokens} "
                    f"tokens, cache row holds {self.max_len}"
                )
            self.waiting.popleft()
            row = self.n_active
            self.n_active += 1
            self.rows[row] = req
            req.row = row
            req.t_admit = now
            self.cache.reset(row)
            admitted.append(req)
        return admitted

    def retire(self, req: Request) -> None:
        """Free a row by swapping the last active row into it."""
        row, last = req.row, self.n_active - 1
        if row != last:
            moved = self.rows[last]
            self.cache.move_row(last, row)
            self.rows[row] = moved
            moved.row = row
        else:
            self.cache.reset(row)
        self.rows[last] = None
        self.n_active -= 1
        req.row = -1

    # ---- forward passes ----------------------------------------------------

    def prefill(self, req: Request, now: float) -> None:
        """Whole prompt in one pass; the logits also give the first token."""
        rows = slice(req.row, req.row + 1)
        ids = req.input_ids.to(self.device)
        positions = torch.arange(req.prompt_len, device=self.device)[None]
        logits = self.model(ids, self.cache, rows, positions)
        token = int(logits[:, -1].argmax(-1).item())
        req.t_first_token = now
        self._record(req, token, now)

    def decode_step(self) -> None:
        """One token for every live row, each at its own position."""
        rows = slice(0, self.n_active)
        live = self.rows[:self.n_active]
        ids = torch.tensor([[r.out_ids[-1]] for r in live], device=self.device)
        # a row's next position is exactly how many tokens it has cached
        positions = torch.tensor([[self.cache.lengths[r.row]] for r in live],
                                 device=self.device)
        logits = self.model(ids, self.cache, rows, positions)
        tokens = logits[:, -1].argmax(-1).tolist()

        now = time.perf_counter()
        for req, token in zip(live, tokens):
            self._record(req, int(token), now)

    def _record(self, req: Request, token: int, now: float) -> None:
        if token in self.stop_ids:
            req.finish = "eos"
            req.t_done = now
            return
        req.out_ids.append(token)
        if req.n_generated >= req.max_new_tokens:
            req.finish = "length"
            req.t_done = now

    # ---- the loop ----------------------------------------------------------

    def run(self, requests: list[Request], *, progress: bool = False) -> RunResult:
        for req in requests:
            req.out_ids.clear()
            req.t_admit = req.t_first_token = req.t_done = None
            req.finish = ""
            self.waiting.append(req)

        step_t: list[float] = []
        step_batch: list[int] = []
        step_ms: list[float] = []
        step_kind: list[str] = []

        sync(self.device)
        t_start = time.perf_counter()
        for req in requests:
            req.arrival = t_start

        while self.waiting or self.n_active:
            for req in self.admit(time.perf_counter()):
                t0 = time.perf_counter()
                with torch.no_grad():
                    self.prefill(req, t0)
                sync(self.device)
                t1 = time.perf_counter()
                step_t.append(t0 - t_start)
                step_batch.append(self.n_active)
                step_ms.append((t1 - t0) * 1000)
                step_kind.append("prefill")

            # a one-token request can finish during prefill
            for req in [r for r in self.rows[:self.n_active] if r and r.finish]:
                self.retire(req)
            if self.n_active == 0:
                continue

            t0 = time.perf_counter()
            with torch.no_grad():
                self.decode_step()
            sync(self.device)
            t1 = time.perf_counter()
            step_t.append(t0 - t_start)
            step_batch.append(self.n_active)
            step_ms.append((t1 - t0) * 1000)
            step_kind.append("decode")

            for req in [r for r in self.rows[:self.n_active] if r and r.finish]:
                self.retire(req)

            if progress:
                done = sum(1 for r in requests if r.finish)
                print(f"\r  step {len(step_ms):5d} | batch {self.n_active:3d} | "
                      f"done {done}/{len(requests)}", end="", flush=True)

        sync(self.device)
        wall = time.perf_counter() - t_start
        if progress:
            print()

        return RunResult(
            policy=self.policy, requests=requests, wall=wall,
            step_t=step_t, step_batch=step_batch, step_ms=step_ms,
            step_kind=step_kind, cache_bytes=self.cache.bytes_reserved,
        )


# ---- convenience for the demo ----------------------------------------------


def load(device=None, dtype=DTYPE):
    """Tokenizer, config and the ragged model, loaded without a 24 GB spike."""
    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = load_config()
    with torch.device("meta"):
        model = Qwen3Continuous(cfg)
    model.load_state_dict(load_weights(), strict=True, assign=True)
    return tok, cfg, model.eval().to(device, dtype), device


def encode(tok, prompts, device, max_new_tokens=MAX_NEW_TOKENS) -> list[Request]:
    requests = []
    for i, prompt in enumerate(prompts):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        requests.append(Request(rid=i, input_ids=ids, prompt=prompt,
                                max_new_tokens=max_new_tokens))
    return requests


def demo(max_batch=MAX_BATCH, max_new_tokens=MAX_NEW_TOKENS) -> None:
    tok, cfg, model, device = load()
    requests = encode(tok, PROMPTS, device, max_new_tokens)
    max_len = min(MAX_LEN, max(r.prompt_len + r.max_new_tokens for r in requests))
    engine = Engine(model, cfg, stop_ids=tok.all_special_ids,
                    max_batch=max_batch, max_len=max_len)
    print(f"{len(requests)} requests, {max_batch} rows\n")

    result = engine.run(requests, progress=True)
    for r in result.requests:
        print(f"> {r.prompt}\n{tok.decode(r.out_ids)}\n")
    ttft = sorted(r.ttft for r in result.requests)
    print(f"{result.wall:.2f}s | {result.out_tokens} tok | "
          f"{result.throughput:.2f} tok/s | mean decode batch "
          f"{result.mean_batch:.2f} | ttft min {ttft[0]:.2f}s max {ttft[-1]:.2f}s")


# ---- self-check ------------------------------------------------------------


def _reference_generate(cfg, model_kv, ids, n_new, stop_ids):
    """Ground truth: the verified batch-1 path in model/qwen_kv.py."""
    from model.qwen_kv import KVCache

    cache = KVCache(cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                    cfg["head_dim"], max_len=256, dtype=torch.float32, device="cpu")
    out: list[int] = []
    with torch.no_grad():
        step = ids
        for _ in range(n_new):
            token = int(model_kv(step, cache)[:, -1].argmax(-1).item())
            if token in stop_ids:
                break
            out.append(token)
            step = torch.tensor([[token]])
    return out


def selftest(seed: int = 0, max_batch: int = 2) -> bool:
    """Scheduling must not change a single token.

    Ragged rows, mid-flight admission and swap-eviction all have to reproduce
    the batch-1 path in model/qwen_kv.py exactly. Random weights are enough:
    this checks the bookkeeping, not the arithmetic model/qwen.py already
    verified against HuggingFace.
    """
    from model.qwen_kv import Qwen3 as Qwen3KV

    torch.manual_seed(seed)
    cfg = dict(TINY)

    ref_model = Qwen3Continuous(cfg).eval().float()
    kv_model = Qwen3KV(cfg).eval().float()
    # same names as the checkpoint, so this is also a load_state_dict smoke test
    kv_model.load_state_dict(ref_model.state_dict(), strict=True)
    print(f"state_dict transfers to model/qwen_kv.Qwen3: {len(ref_model.state_dict())} tensors")

    stop_ids: set[int] = set()      # no early exit, so lengths are exactly as asked
    specs = [(11, 24), (5, 6), (17, 15), (3, 9), (8, 20)]   # ragged prompts and lengths
    prompts = [torch.randint(0, cfg["vocab_size"], (1, p)) for p, _ in specs]
    expected = [
        _reference_generate(cfg, kv_model, ids, n, stop_ids)
        for ids, (_, n) in zip(prompts, specs)
    ]

    requests = [
        Request(rid=i, input_ids=ids, max_new_tokens=n, label=f"r{i}")
        for i, (ids, (_, n)) in enumerate(zip(prompts, specs))
    ]
    engine = Engine(ref_model, cfg, stop_ids=stop_ids, max_batch=max_batch, max_len=64)
    result = engine.run(requests)

    ok = True
    for req, want in zip(result.requests, expected):
        same = req.out_ids == want
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} r{req.rid} prompt={req.prompt_len:3d} "
              f"generated={req.n_generated:3d}")
        if not same:
            print(f"     want {want}\n     got  {req.out_ids}")
    print(f"     {result.out_tokens} tokens, {len(result.step_ms)} forwards, "
          f"mean decode batch {result.mean_batch:.2f}")

    print("\nPASS" if ok else "\nFAIL")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="equivalence check on tiny random weights, no download")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest(args.seed, max(2, args.max_batch // 2)) else 1)
    demo(args.max_batch, args.max_new_tokens)


if __name__ == "__main__":
    main()

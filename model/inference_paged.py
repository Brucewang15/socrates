"""Continuous batching over a paged KV cache.

    ENGINE=paged uv run -m model.inference_paged

Same scheduler as inference_cont -- admit, sequential prefill, one batched decode
step, retire and refill -- with the cache underneath swapped for blocks. Three
things change and nothing else does:

  admit   reserves a request's worst case, ceil((prompt + MAX_NEW_TOKENS) / 16)
          blocks, before its prefill. With MAX_BATCH rows and a pool sized
          MAX_BATCH x MAX_LEN, that reservation cannot fail, so there is no
          preemption to write. Allocating lazily and admitting past 18 rows is
          the follow-up; see docs/day_8_docs.md.
  decode  has no read window. Each row's extent is its own length, read on the
          device, so the launch shape depends only on the batch width.
  retire  frees the row's blocks, then compacts by copying one table row.

Request, the exceptions, the prompt list, the limits and the Prometheus counters
are imported from inference_cont rather than redefined: registering a counter
name twice raises, and the server catches cont.QueueFull/cont.TooLong whichever
engine is running.
"""

import os
import time
from collections import deque

import torch

from model.inference_cont import (
    COMPILE,
    DEVICE,
    DTYPE,
    IN_TOKENS,
    MAX_BATCH,
    MAX_LEN,
    MAX_NEW_TOKENS,
    MAX_QUEUE,
    OUT_TOKENS,
    PROMPTS,
    QueueFull,
    Request,
    TooLong,
)
from model.kernels import require_cuda_kernel
from model.paging import BlockAllocator, blocks_for
from model.qwen.qwen_kv_paged import MODEL_ID, PagedKVCache, Qwen3Paged, load_config, load_weights

# 16 slots x 8 kv heads x 128 dims x 2 B = 32 KB per block per layer for K, the
# same for V. The kernel is specialised for 16; see model/kernels/csrc.
KV_BLOCK = int(os.getenv("KV_BLOCK", "16"))


class Engine:
    def __init__(self, model: Qwen3Paged | None = None, tok=None, device: str = DEVICE,
                 dtype=DTYPE, compile: bool = COMPILE, max_batch: int = MAX_BATCH,
                 max_len: int = MAX_LEN, max_new_tokens: int = MAX_NEW_TOKENS,
                 block_size: int = KV_BLOCK, num_blocks: int | None = None):
        """Defaults serve the real checkpoint. Tests pass a tiny model and a stub
        tokenizer instead, which is the only reason the knobs are arguments."""
        self.device, self.max_batch = device, max_batch
        self.max_len, self.max_new_tokens = max_len, max_new_tokens
        if device.startswith("cuda"):
            # here, not on the first decode step: see require_cuda_kernel
            require_cuda_kernel()

        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tok = tok
        if model is None:
            cfg = load_config()
            with torch.device("meta"):
                model = Qwen3Paged(cfg)
            model.load_state_dict(load_weights(), assign=True)
            model = model.eval().to(device)
        self.model = model
        self.cfg = cfg = model.cfg

        # Worst case for every row at once, so admission never runs dry. That is
        # 2304 blocks here, the same 5.3 GB the dense cache held -- this change is
        # about bytes read, not bytes held.
        per_row = blocks_for(max_len, block_size)
        num_blocks = num_blocks or max_batch * per_row
        self.allocator = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(max_batch, cfg["num_hidden_layers"],
                                  cfg["num_key_value_heads"], cfg["head_dim"],
                                  num_blocks=num_blocks, block_size=block_size,
                                  max_blocks_per_row=per_row,
                                  dtype=next(model.parameters()).dtype, device=device)
        self.model.attach_cache(self.cache)

        self.step_ids = torch.zeros((max_batch, 1), dtype=torch.long, device=device)
        self.step_pos = torch.zeros((max_batch, 1), dtype=torch.long, device=device)

        # Decode compiled, prefill eager: inference_cont explains why.
        self.eager_model = self.model
        self.decode_model = self.model
        if compile and device.startswith("cuda"):
            self.decode_model = torch.compile(self.model, mode="reduce-overhead",
                                              dynamic=False)
            print("paged decode path compiled (mode=reduce-overhead)", flush=True)

        self.loop = None
        self.rows: list[Request | None] = [None] * max_batch
        self.n_active = 0
        self.pending: deque[Request] = deque()
        self.warmup()

    @torch.no_grad()
    def warmup(self) -> None:
        """One graph per batch width, and that is all of them.

        The dense engine needed width x window bucket = 72 graphs, because the
        window was part of the shape. Here nothing about a row's length reaches
        the shape, so 18 widths cover every context. No row owns a block yet, so
        every write goes to the null page and nothing needs resetting after.
        """
        if self.decode_model is self.model:
            return
        t0 = time.perf_counter()
        self.step_ids.zero_()
        for width in range(1, self.max_batch + 1):
            for step in range(3):          # capture happens past the first call
                self.step_pos[:width].fill_(step)
                self.decode_model(self.step_ids[:width], slice(0, width),
                                  self.step_pos[:width])
        torch.cuda.synchronize()
        print(f"paged decode graphs warm for 1..{self.max_batch} rows "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)

    def submit(self, prompt: str, stream: bool = False) -> Request:
        if len(self.pending) >= MAX_QUEUE:
            raise QueueFull(f"{len(self.pending)} requests already waiting")
        req = Request(prompt)
        if stream:
            import asyncio
            req.stream = asyncio.Queue()
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        req.ids = self.tok(text, return_tensors="pt").input_ids
        if req.ids.shape[1] + self.max_new_tokens > self.max_len:
            raise TooLong(f"prompt is {req.ids.shape[1]} tokens; with "
                          f"{self.max_new_tokens} new it exceeds the {self.max_len}-token row")
        self.pending.append(req)
        return req

    def reserve(self, req: Request) -> int:
        """Blocks for every slot this request can ever write.

        The last decode step writes position prompt + MAX_NEW_TOKENS - 1, so the
        row needs prompt + MAX_NEW_TOKENS slots; submit() already refused
        anything that would not fit in one row.
        """
        return blocks_for(req.ids.shape[1] + self.max_new_tokens, self.cache.block_size)

    @torch.no_grad()
    def prefill(self, req: Request, row: int) -> None:
        ids = req.ids.to(self.device)
        positions = torch.arange(ids.shape[1], device=self.device)[None]
        logits = self.eager_model(ids, slice(row, row + 1), positions)
        self.cache.lengths[row] = ids.shape[1]
        IN_TOKENS.inc(ids.shape[1])
        self.record(req, int(logits[:, -1].argmax(-1)))

    def admit(self) -> None:
        while self.pending and self.n_active < self.max_batch:
            req = self.pending.popleft()
            if req.cancelled:
                continue
            row = self.n_active
            self.cache.assign(row, self.allocator.alloc(self.reserve(req)))
            self.rows[row] = req
            req.row = row
            req.admitted = time.perf_counter()
            self.n_active += 1
            self.prefill(req, row)

    @torch.no_grad()
    def decode_step(self) -> None:
        live = self.rows[:self.n_active]
        n = self.n_active
        self.step_ids[:n].copy_(
            torch.tensor([[r.output[-1]] for r in live], dtype=torch.long))
        pos = [self.cache.lengths[r.row] for r in live]
        self.step_pos[:n].copy_(torch.tensor(pos, dtype=torch.long)[:, None])
        logits = self.decode_model(self.step_ids[:n], slice(0, n), self.step_pos[:n])
        for req in live:
            self.cache.lengths[req.row] += 1
        # tolist() copies off the graph's output buffer before the next replay
        for req, token in zip(live, logits[:, -1].argmax(-1).tolist()):
            self.record(req, int(token))

    def emit(self, req: Request, item: int | None) -> None:
        if req.stream is not None:
            self.loop.call_soon_threadsafe(req.stream.put_nowait, item)

    def record(self, req: Request, token: int) -> None:
        now = time.perf_counter()
        if not req.first_token:
            req.first_token = now
        if token in self.tok.all_special_ids or len(req.output) >= self.max_new_tokens:
            req.done = True
            req.finished = now
            self.emit(req, None)
        else:
            req.output.append(token)
            OUT_TOKENS.inc()
            self.emit(req, token)

    def retire(self, req: Request) -> None:
        row, last = req.row, self.n_active - 1
        self.allocator.free(self.cache.release(row))
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

    def sweep(self) -> None:
        for req in [r for r in self.rows[:self.n_active] if r and (r.done or r.cancelled)]:
            self.retire(req)

    def run(self) -> None:
        while self.pending or self.n_active:
            self.admit()
            self.sweep()
            if self.n_active:
                self.decode_step()
            self.sweep()


def main() -> None:
    engine = Engine()
    reqs = [engine.submit(p) for p in PROMPTS]
    print(f"{len(reqs)} requests, {engine.max_batch} rows, "
          f"{engine.allocator.num_blocks} blocks of {engine.cache.block_size}\n")
    t0 = time.perf_counter()
    engine.run()
    wall = time.perf_counter() - t0
    for r in reqs:
        print(f"> {r.prompt}\n{engine.tok.decode(r.output)}\n")
    total = sum(len(r.output) for r in reqs)
    print(f"{wall:.2f}s | {total} tok | {total / wall:.2f} tok/s")


if __name__ == "__main__":
    main()

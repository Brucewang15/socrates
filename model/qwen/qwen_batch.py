"""
Ragged-batch forward pass plus a continuous-batching scheduler.

    uv run -m model.qwen.qwen_batch            # equivalence check, tiny random weights
    uv run -m model.qwen.qwen_batch --help     # knobs

Day 2 code is untouched. RMSNorm, MLP, rotate_half/apply_rope and the weight
loaders are imported from model/qwen/qwen_kv.py; only the parts that genuinely had
to change to serve several sequences at once are rewritten here. Module
attribute names still match the checkpoint, so load_state_dict(strict=True)
works exactly as before.

Three things separate this from the batch-1 cache in model/qwen/qwen_kv.py:

  1. the cache is [max_batch, max_len, n_kv_heads, head_dim] and every row
     carries its own length, so rows are *ragged* -- no padding to the longest;
  2. positions and the mask are derived per row from that length, so a row
     never sees another row's tokens, nor its own unwritten slots;
  3. active rows stay packed in 0..n_active-1. Retiring a row swaps the last
     active row into the hole, so admission and eviction never disturb the
     other sequences and every cache read stays a view rather than a gather.

Point 3 is what makes continuous batching cheap. The scheduler below then only
has to decide, each step, who is allowed in -- see Engine.admit.

    static      one wave at a time: nobody joins until the whole wave drains,
                so a 500-token request holds 20-token requests hostage.
    continuous  every step re-admits into whatever rows are free.

Both policies run the identical kernels through the identical cache. The only
difference is the admission rule, which is the point: any gap you measure
between them is scheduling, not arithmetic.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from model.qwen.qwen_kv import MLP, MODEL_ID, RMSNorm, apply_rope, load_config, load_weights
from torch import nn

__all__ = [
    "MODEL_ID",
    "BatchedKVCache",
    "Engine",
    "Qwen3Batch",
    "Request",
    "RunResult",
    "load_config",
    "load_weights",
    "pick_device",
]


def pick_device(name: str | None = None) -> torch.device:
    """cuda on the g6e box, mps on the laptop, cpu as the fallback."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Make timing honest: the queue is async on both cuda and mps."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def rope_tables(cfg: dict, positions: torch.Tensor):
    """cos, sin of shape [rows, T, head_dim] for arbitrary per-row positions.

    model/qwen/qwen_kv.py builds one table for a contiguous [offset, offset+T) span
    because it only ever has one sequence. Here every row sits at a different
    absolute position, so the table is indexed by the position of each token.
    """
    hd, base = cfg["head_dim"], cfg["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=positions.device).float() / hd))
    angles = positions.float()[..., None] * inv_freq        # [rows, T, hd/2]
    angles = torch.cat([angles, angles], dim=-1)            # duplicated to full width
    return angles.cos(), angles.sin()


class BatchedKVCache:
    """Post-RoPE K and V for several ragged sequences, one buffer pair per layer.

    Layout per layer: [max_batch, max_len, n_kv_heads, head_dim].

    Rows 0..n_active-1 are live and each has its own length. The engine owns
    which request sits in which row; the cache only knows lengths.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, *,
                 max_batch: int, max_len: int,
                 dtype=torch.bfloat16, device="cpu"):
        shape = (max_batch, max_len, n_kv_heads, head_dim)
        # one buffer per layer; [x] * n would alias a single tensor n times
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.lengths = [0] * max_batch
        self.max_batch = max_batch
        self.max_len = max_len
        itemsize = torch.empty((), dtype=dtype).element_size()
        # the day-1 number: 2 (K and V) * layers * kv_heads * head_dim * dtype
        self.bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * itemsize
        self.bytes_reserved = self.bytes_per_token * max_batch * max_len

    def reset(self, row: int) -> None:
        """Hand a row to a new request. Stale bytes stay, the mask hides them."""
        self.lengths[row] = 0

    def advance(self, rows: slice, n: int) -> None:
        for row in range(rows.start, rows.stop):
            self.lengths[row] += n

    def move_row(self, src: int, dst: int) -> None:
        """Compact: relocate a live row so the active block stays packed."""
        n = self.lengths[src]
        for k, v in zip(self.keys, self.values):
            k[dst, :n] = k[src, :n]
            v[dst, :n] = v[src, :n]
        self.lengths[dst] = n
        self.lengths[src] = 0

    def append(self, layer_idx: int, rows: slice, k, v, offsets: torch.Tensor, seq_len: int):
        """Write this step's k, v then return the live span for these rows.

        k, v: [rows, T, n_kv_heads, head_dim] -- the new tokens only.
        offsets: [rows] absolute write start per row.
        seq_len: how far the returned view has to reach, max(offsets) + T.

        Returns views, not copies, which is the whole reason rows stay packed.
        """
        K, V = self.keys[layer_idx], self.values[layer_idx]
        T = k.shape[1]
        if T == 1:
            # decode: one token per row, each at its own offset -- a scatter
            r = torch.arange(rows.start, rows.stop, device=k.device)
            K[r, offsets] = k[:, 0]
            V[r, offsets] = v[:, 0]
        else:
            # prefill: a single row takes a contiguous block
            start = int(offsets[0])
            K[rows, start:start + T] = k
            V[rows, start:start + T] = v
        return K[rows, :seq_len], V[rows, :seq_len]


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]

        self.q_proj = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, cache, layer_idx, rows, offsets, seq_len, mask):
        B, T, _ = x.shape
        group = self.n_heads // self.n_kv_heads      # query heads sharing one kv head

        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.hd)

        # per-head RMSNorm, then rotate by each token's own absolute position
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        k, v = cache.append(layer_idx, rows, k, v, offsets, seq_len)

        # GQA: head h reads kv head h // group
        k_full = k.repeat_interleave(group, dim=2)
        v_full = v.repeat_interleave(group, dim=2)

        # heads become a batch axis: [rows, heads, T, head_dim]
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        scores = (q @ k_full.transpose(-2, -1)) / self.hd ** 0.5
        # mask is [rows, 1, T, seq_len]: causal within a row, and everything past
        # a row's own length -- other rows' tokens, unwritten slots -- is blocked
        scores = scores.masked_fill(mask, float("-inf"))
        out = scores.softmax(dim=-1) @ v_full

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache, layer_idx, rows, offsets, seq_len, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache,
                               layer_idx, rows, offsets, seq_len, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Batch(nn.Module):
    """Same weights as model/qwen/qwen_kv.Qwen3, ragged batch instead of batch 1."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, input_ids, cache: BatchedKVCache, rows: slice, positions: torch.Tensor):
        """input_ids: [rows, T]. positions: [rows, T] absolute, one per token.

        Prefill is rows of width 1 with positions arange(P); decode is rows of
        width n_active with T = 1 and positions = each row's current length.
        """
        x = self.embed_tokens(input_ids)
        T = input_ids.shape[1]
        cos, sin = rope_tables(self.cfg, positions)

        # positions are contiguous within a row, so column 0 is the write start
        offsets = positions[:, 0]
        seq_len = int(offsets.max().item()) + T

        # query i of a row sits at positions[row, i]; key j sits at j.
        # forbidden when j > positions[row, i]. That one comparison covers the
        # causal rule, the padding, and cross-row isolation all at once.
        k_pos = torch.arange(seq_len, device=input_ids.device)
        mask = (k_pos[None, None, :] > positions[:, :, None])[:, None]

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, rows, offsets, seq_len, mask)
        x = self.norm(x)
        cache.advance(rows, T)
        # unembedding is the embedding matrix transposed (tied)
        return x @ self.embed_tokens.weight.T


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
    policy: str
    requests: list[Request]
    wall: float
    step_t: list[float]              # seconds since start, one per forward
    step_batch: list[int]            # rows live in that forward
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
    """Scheduler + ragged cache. policy='static' or 'continuous'.

    Prefill runs on its own for each admitted request, then merges into the
    decode batch. Interleaving prefill with decode in one forward is the better
    design and is left for later; this one is chosen because it keeps the
    admission logic -- the thing being measured -- readable.
    """

    def __init__(self, model: Qwen3Batch, cfg: dict, *, stop_ids, max_batch: int,
                 max_len: int, policy: str = "continuous"):
        if policy not in ("static", "continuous"):
            raise ValueError(f"policy must be static or continuous, got {policy!r}")
        p = next(model.parameters())
        self.model = model
        self.cfg = cfg
        self.device = p.device
        self.policy = policy
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
        """static: only between waves. continuous: any step, into any free row."""
        if self.policy == "static" and self.n_active > 0:
            return []
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


# ---- self-check ------------------------------------------------------------

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "vocab_size": 128,
}


def _reference_generate(cfg, model_kv, ids, n_new, stop_ids):
    """Ground truth: the verified batch-1 path in model/qwen/qwen_kv.py."""
    from model.qwen.qwen_kv import KVCache

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
    """Ragged batching must reproduce the batch-1 path token for token.

    Random weights are enough: this checks the cache, the mask and the row
    bookkeeping, not the arithmetic model/qwen/qwen.py already verified against HF.
    """
    torch.manual_seed(seed)
    cfg = dict(TINY)

    batch_model = Qwen3Batch(cfg).eval().float()
    from model.qwen.qwen_kv import Qwen3 as Qwen3KV

    kv_model = Qwen3KV(cfg).eval().float()
    # same names as the checkpoint, so this is also a load_state_dict smoke test
    kv_model.load_state_dict(batch_model.state_dict(), strict=True)
    print(f"state_dict transfers to model/qwen/qwen_kv.Qwen3: {len(batch_model.state_dict())} tensors")

    stop_ids: set[int] = set()      # no early exit, so lengths are exactly as asked
    specs = [(11, 24), (5, 6), (17, 15), (3, 9), (8, 20)]   # ragged prompts and lengths

    prompts = [torch.randint(0, cfg["vocab_size"], (1, p)) for p, _ in specs]
    expected = [
        _reference_generate(cfg, kv_model, ids, n, stop_ids)
        for ids, (_, n) in zip(prompts, specs)
    ]

    ok = True
    for policy in ("continuous", "static"):
        requests = [
            Request(rid=i, input_ids=ids, max_new_tokens=n, label=f"r{i}")
            for i, (ids, (_, n)) in enumerate(zip(prompts, specs))
        ]
        engine = Engine(Qwen3Batch(cfg).eval().float(), cfg, stop_ids=stop_ids,
                        max_batch=max_batch, max_len=64, policy=policy)
        engine.model.load_state_dict(batch_model.state_dict(), strict=True)
        result = engine.run(requests)

        for req, want in zip(result.requests, expected):
            same = req.out_ids == want
            ok &= same
            print(f"{'ok  ' if same else 'FAIL'} {policy:10s} r{req.rid} "
                  f"prompt={req.prompt_len:3d} generated={req.n_generated:3d}")
            if not same:
                print(f"     want {want}\n     got  {req.out_ids}")
        print(f"     {policy}: {result.out_tokens} tokens, "
              f"{len(result.step_ms)} forwards, mean decode batch "
              f"{result.mean_batch:.2f}")

    print("\nPASS" if ok else "\nFAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=2,
                    help="small values force staggered admission in the check")
    args = ap.parse_args()
    raise SystemExit(0 if selftest(args.seed, args.max_batch) else 1)


if __name__ == "__main__":
    main()

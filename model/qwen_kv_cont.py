"""
Ragged KV cache and forward pass for continuous batching. Model only.

    uv run -m model.qwen_kv_cont     # per-row equivalence check, tiny random weights

The scheduler is not here. It lives in backend/inference_cont.py, mirroring the
static pair:

    model/qwen_kv_cont.py        ragged cache + forward pass   (this file)
    backend/inference_cont.py    the continuous scheduler

    model/qwen_kv_seq.py         padded cache + forward pass
    backend/inference_static.py  the wave loop

Day 2 code is untouched. RMSNorm, MLP, rotate_half/apply_rope and the weight
loaders are imported from model/qwen_kv.py; only the parts that genuinely had
to change to serve several sequences at once are rewritten here. Module
attribute names still match the checkpoint, so load_state_dict(strict=True)
works exactly as before.

Three things separate this from the batch-1 cache in model/qwen_kv.py:

  1. the cache is [max_batch, max_len, n_kv_heads, head_dim] and every row
     carries its own length, so rows are *ragged* -- no padding to the longest;
  2. positions and the mask are derived per row from that length, so a row
     never sees another row's tokens, nor its own unwritten slots;
  3. active rows stay packed in 0..n_active-1. move_row lets a caller close a
     hole by relocating one row, so every cache read stays a view rather than a
     gather.

Point 3 is what makes continuous batching cheap: eviction costs one row, not the
batch, so a scheduler can refill a seat the instant it frees.

The caller owns policy. It says which rows it is running -- a contiguous slice --
and the absolute position of every token it feeds; everything else follows.
"""

from __future__ import annotations

import argparse

import torch
from model.qwen_kv import MLP, MODEL_ID, RMSNorm, apply_rope, load_config, load_weights
from torch import nn

__all__ = [
    "MODEL_ID",
    "TINY",
    "BatchedKVCache",
    "Qwen3Continuous",
    "load_config",
    "load_weights",
    "pick_device",
    "rope_tables",
    "sync",
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

    model/qwen_kv.py builds one table for a contiguous [offset, offset+T) span
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


class Qwen3Continuous(nn.Module):
    """Same weights as model/qwen_kv.Qwen3, ragged batch instead of batch 1."""

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


def _reference_generate(cfg, model_kv, ids, n_new):
    """Ground truth: the verified batch-1 path in model/qwen_kv.py."""
    from model.qwen_kv import KVCache

    cache = KVCache(cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                    cfg["head_dim"], max_len=256, dtype=torch.float32, device="cpu")
    out: list[int] = []
    with torch.no_grad():
        step = ids
        for _ in range(n_new):
            token = int(model_kv(step, cache)[:, -1].argmax(-1).item())
            out.append(token)
            step = torch.tensor([[token]])
    return out


def selftest(seed: int = 0) -> bool:
    """A ragged batch must produce what the batch-1 path produces, per row.

    Model and cache only -- no scheduler. Rows are prefilled at different
    lengths and then decoded together, so this exercises the per-row positions,
    the mask, and the ragged writes. Scheduling is checked separately by
    `uv run -m backend.inference_cont --selftest`.

    Random weights are enough: the arithmetic is already verified against
    HuggingFace by model/qwen.py.
    """
    from model.qwen_kv import Qwen3 as Qwen3KV

    torch.manual_seed(seed)
    cfg = dict(TINY)

    model = Qwen3Continuous(cfg).eval().float()
    kv_model = Qwen3KV(cfg).eval().float()
    # same names as the checkpoint, so this is also a load_state_dict smoke test
    kv_model.load_state_dict(model.state_dict(), strict=True)
    print(f"state_dict transfers to model/qwen_kv.Qwen3: {len(model.state_dict())} tensors")

    specs = [(11, 24), (5, 6), (17, 15), (3, 9), (8, 20)]   # (prompt_len, tokens)
    prompts = [torch.randint(0, cfg["vocab_size"], (1, p)) for p, _ in specs]
    expected = [_reference_generate(cfg, kv_model, ids, n)
                for ids, (_, n) in zip(prompts, specs)]

    rows = len(specs)
    cache = BatchedKVCache(cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                           cfg["head_dim"], max_batch=rows, max_len=64,
                           dtype=torch.float32, device="cpu")
    got: list[list[int]] = [[] for _ in specs]

    with torch.no_grad():
        # prefill each row at its own length -- this is what makes rows ragged
        for row, ids in enumerate(prompts):
            positions = torch.arange(ids.shape[1])[None]
            logits = model(ids, cache, slice(row, row + 1), positions)
            got[row].append(int(logits[:, -1].argmax(-1).item()))

        # then decode every row together, each from its own position
        for _ in range(max(n for _, n in specs) - 1):
            ids = torch.tensor([[g[-1]] for g in got])
            positions = torch.tensor([[cache.lengths[r]] for r in range(rows)])
            logits = model(ids, cache, slice(0, rows), positions)
            tokens = logits[:, -1].argmax(-1).tolist()
            for r, token in enumerate(tokens):
                if len(got[r]) < specs[r][1]:
                    got[r].append(int(token))

    ok = True
    for r, (want, mine) in enumerate(zip(expected, got)):
        same = mine == want
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} row {r} prompt={specs[r][0]:3d} "
              f"generated={len(mine):3d}")
        if not same:
            print(f"     want {want}\n     got  {mine}")

    # eviction relies on move_row relocating a live row untouched
    n = cache.lengths[2]
    before_k = cache.keys[0][2, :n].clone()
    before_v = cache.values[0][2, :n].clone()
    cache.move_row(2, 0)
    moved = (torch.equal(cache.keys[0][0, :n], before_k)
             and torch.equal(cache.values[0][0, :n], before_v)
             and cache.lengths[0] == n)
    ok &= moved
    print(f"{'ok  ' if moved else 'FAIL'} move_row relocates {n} tokens intact")

    print("\nPASS" if ok else "\nFAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(0 if selftest(args.seed) else 1)


if __name__ == "__main__":
    main()

"""Paged KV cache for continuous batching. Model only -- no scheduler.

The dense cache (qwen_kv_cont) gives every row its own [max_len] rectangle and
reads one window for all of them, sized to the longest. Here the cache is a pool
of fixed-size blocks per layer, and each row owns a list of them:

    block_table[row, i]  = physical block holding the row's slots i*BS .. i*BS+BS-1
    slot t of row r      = pool[block_table[r, t // BS], :, t % BS]

Decode attention walks a row's own table up to its own length (see
model/kernels), so a short row no longer pays for the long one beside it.

Everything the scheduler decides -- which blocks a row owns, how long it is --
lives on the host and is copied into block_table between steps. Nothing inside a
forward reads a value back from the device, which is what keeps it capturable.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model.kernels import paged_attention_decode
from model.qwen.qwen_kv_cont import (  # noqa: F401  (re-exported for the engine)
    MLP,
    MODEL_ID,
    RMSNorm,
    apply_rope,
    load_config,
    load_weights,
    rope_tables,
)


class PagedLayerCache(nn.Module):
    """One layer's K and V pools, [num_blocks + 1, n_kv_heads, block_size, head_dim].

    Heads before slots so that one (block, head) is a contiguous tile -- 16 slots
    x 128 dims x 2 B = 4 KB -- which is what the kernel reads with 16-byte loads.
    The +1 is block 0, the null page. Buffers for the reason given in KVCache:
    module state can be written under a CUDA graph, graph inputs cannot.
    """

    def __init__(self, shape, dtype, device):
        super().__init__()
        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device),
                             persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device),
                             persistent=False)


class PagedKVCache(nn.Module):
    """Block pools per layer, one block table for all of them.

    Every layer uses the same table: a row's slot t is in the same physical block
    in all 36 layers, so the indirection is computed once per step, not per layer.

    move_row is the change that falls out for free. In the dense cache it copied
    up to 2048 slots x 36 layers x K and V to keep live rows packed at 0..n-1; here
    it copies one 128-entry row of the table and leaves the KV where it is.
    """

    def __init__(self, max_batch: int, n_layers: int, n_kv_heads: int, head_dim: int,
                 num_blocks: int, block_size: int = 16, max_blocks_per_row: int = 128,
                 dtype=torch.bfloat16, device="mps"):
        super().__init__()
        shape = (num_blocks + 1, n_kv_heads, block_size, head_dim)
        self.layers = nn.ModuleList(
            PagedLayerCache(shape, dtype, device) for _ in range(n_layers))
        # int32 because that is what the kernel indexes with. Zero is the null
        # page, so a fresh table sends every row's writes somewhere harmless.
        self.register_buffer(
            "block_table",
            torch.zeros((max_batch, max_blocks_per_row), dtype=torch.int32, device=device),
            persistent=False)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_blocks_per_row = max_blocks_per_row
        # host-side, for the same reason as KVCache.lengths
        self.lengths = [0] * max_batch
        self.row_blocks: list[list[int]] = [[] for _ in range(max_batch)]

    def assign(self, row: int, blocks: list[int]) -> None:
        """Hand `blocks` to an empty row. One small H2D copy, between steps."""
        if self.row_blocks[row]:
            raise ValueError(f"row {row} still owns {len(self.row_blocks[row])} blocks")
        if len(blocks) > self.max_blocks_per_row:
            raise ValueError(f"{len(blocks)} blocks > table width {self.max_blocks_per_row}")
        self.row_blocks[row] = list(blocks)
        self.lengths[row] = 0
        self.block_table[row].zero_()
        if blocks:
            self.block_table[row, :len(blocks)].copy_(
                torch.tensor(blocks, dtype=torch.int32))

    def release(self, row: int) -> list[int]:
        """Take a row's blocks back; the caller returns them to the allocator."""
        blocks, self.row_blocks[row] = self.row_blocks[row], []
        self.lengths[row] = 0
        self.block_table[row].zero_()
        return blocks

    def move_row(self, src: int, dst: int) -> None:
        """Relocate a live row so the active block stays packed at 0..n-1."""
        if self.row_blocks[dst]:
            raise ValueError(f"row {dst} still owns blocks; release it first")
        self.block_table[dst].copy_(self.block_table[src])
        self.block_table[src].zero_()
        self.row_blocks[dst], self.row_blocks[src] = self.row_blocks[src], []
        self.lengths[dst], self.lengths[src] = self.lengths[src], 0

    def write(self, layer_idx: int, rows: slice, k, v, positions) -> None:
        """Store this step's k, v at each row's own positions.

        k, v:      [rows, T, n_kv_heads, head_dim]
        positions: [rows, T] absolute position of each token

        The slot mapping is computed on the device from the table, so it is part
        of the captured graph rather than a host-built index per step.
        """
        layer = self.layers[layer_idx]
        bt = self.block_table[rows]                                    # [R, W]
        blk = bt.gather(1, positions // self.block_size).long()        # [R, T]
        off = positions % self.block_size                              # [R, T]
        # Advanced indices either side of a slice put their dims first, so the
        # destination is [R, T, n_kv_heads, head_dim] -- exactly k's shape.
        layer.k[blk, :, off] = k
        layer.v[blk, :, off] = v


class PagedAttention(nn.Module):
    """Same projections and parameter names as qwen_kv_cont.Attention."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, cache: PagedKVCache, layer_idx, rows, positions,
                seq_lens):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.hd)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        cache.write(layer_idx, rows, k, v, positions)

        if T == 1:
            # Decode: one query per row against that row's own pages, up to its
            # own length. This is the read paging exists to shrink.
            layer = cache.layers[layer_idx]
            out = paged_attention_decode(
                q[:, 0], layer.k, layer.v, cache.block_table[rows], seq_lens,
                self.scale)                                            # [B, Hq, D]
            out = out.reshape(B, 1, self.n_heads * self.hd)
        else:
            # Prefill: a prompt starts at position 0 and has no prior context --
            # there is no prefix sharing -- so every key it can see is one it is
            # computing right now. Attend to those directly and causally; the
            # pages are written for the decode steps that follow, never read here.
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=True)                       # [B, Hq, T, D]
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.o_proj(out)


class PagedBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = PagedAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache, layer_idx, rows, positions, seq_lens):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx,
                               rows, positions, seq_lens)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Paged(nn.Module):
    """Qwen3 over a PagedKVCache. Parameter names match the checkpoint exactly."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(PagedBlock(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.cache: PagedKVCache | None = None

    def attach_cache(self, cache: PagedKVCache) -> None:
        """Registered as a child module, for the reason in qwen_kv_cont.attach_cache."""
        self.cache = cache

    def forward(self, input_ids, rows: slice, positions):
        """input_ids and positions are [rows, T]; rows selects cache rows.

        There is no `end` any more. The dense model needed one because every row
        read the same window; here each row's extent is its own seq_len, and that
        is derived on the device from `positions` -- the last token's position
        plus one, i.e. including the slot written this step. The launch shape
        depends only on the batch width, so one graph per width covers every
        context length.
        """
        cache = self.cache
        seq_lens = (positions[:, -1] + 1).to(torch.int32)
        x = self.embed_tokens(input_ids)
        cos, sin = rope_tables(self.cfg, positions)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, rows, positions, seq_lens)
        x = self.norm(x)
        return x @ self.embed_tokens.weight.T

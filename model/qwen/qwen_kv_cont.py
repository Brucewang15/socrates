"""
Ragged KV cache for continuous batching. Model only -- no scheduler.

Every row carries its own length, so rows can sit at different positions. The
caller says which rows it is running and the absolute position of each token
it is feeding; everything else follows from that.
"""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qwen3-4b.json"
MODEL_ID = "Qwen/Qwen3-4B"


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def load_weights(model_id: str = MODEL_ID) -> dict:
    """Checkpoint tensors, keyed with the leading 'model.' stripped."""

    index = json.loads(Path(hf_hub_download(model_id, "model.safetensors.index.json")).read_text())
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(hf_hub_download(model_id, shard)))
    return {k.removeprefix("model."): v for k, v in sd.items()}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def rope_tables(cfg: dict, positions):
    """cos, sin of shape [rows, T, head_dim] for arbitrary absolute positions."""
    hd, base = cfg["head_dim"], cfg["rope_theta"]
    # theta_k = base ** (-2k / hd), one per dimension pair
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=positions.device).float() / hd))
    angles = positions.float()[..., None] * inv_freq
    # Duplicated to full width so apply_rope is elementwise.
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    """x: [batch, seq, heads, head_dim]."""
    # tables are built in float32; match x so bf16 inference doesn't upcast
    cos = cos[:, :, None, :].to(x.dtype)
    sin = sin[:, :, None, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class KVCache:
    """Post-RoPE keys and values, one buffer per layer, one row per sequence.

    Rows are ragged: lengths[r] is how many tokens row r has written. A row is
    handed to a new request by setting its length back to 0.
    """

    def __init__(self, max_batch: int, n_layers: int, n_kv_heads: int, head_dim: int,
                 max_len: int = 2048, dtype=torch.bfloat16, device="mps"):
        shape = (max_batch, max_len, n_kv_heads, head_dim)
        # one buffer per layer, [x] * n would ref a single tensor n times
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.lengths = [0] * max_batch

    def reset(self, row: int) -> None:
        self.lengths[row] = 0

    def move_row(self, src: int, dst: int) -> None:
        """Relocate a live row so the active block stays packed at 0..n-1."""
        n = self.lengths[src]
        for k, v in zip(self.keys, self.values):
            k[dst, :n] = k[src, :n]
            v[dst, :n] = v[src, :n]
        self.lengths[dst], self.lengths[src] = n, 0

    def append(self, layer_idx: int, rows: slice, k, v, positions):
        """Write k, v at each row's own positions, return every row's live span.

        k, v:      [rows, T, n_kv_heads, head_dim], the new tokens only
        positions: [rows, T], absolute position of each of those tokens
        """
        K, V = self.keys[layer_idx], self.values[layer_idx]
        r = torch.arange(rows.start, rows.stop, device=k.device)[:, None]
        K[r, positions] = k
        V[r, positions] = v
        end = int(positions.max()) + 1
        for i, row in enumerate(range(rows.start, rows.stop)):
            self.lengths[row] = int(positions[i, -1]) + 1
        return K[rows, :end], V[rows, :end]


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]

        # All heads' q_proj and k_proj are stored together, row 0-127 = head 0s, 128-255 = head 1s, etc.
        self.q_proj = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])

    def forward(self, x, cos, sin, cache, layer_idx, rows, positions):
        B, T, _ = x.shape
        group = self.n_heads // self.n_kv_heads          # 4 query heads share one kv head

        # project, then split the packed output into heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.hd)

        # per-head RMSNorm, then rotate by position
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        
        k, v = cache.append(layer_idx, rows, k, v, positions)

        S = k.shape[1]

        # give every query head its group's k and v
        k_full = torch.empty(B, S, self.n_heads, self.hd, dtype=k.dtype, device=k.device)
        v_full = torch.empty_like(k_full)
        for h in range(self.n_heads):
            k_full[:, :, h] = k[:, :, h // group]
            v_full[:, :, h] = v[:, :, h // group]

        # heads become a batch axis: [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)
    
        # every query against every key
        scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.hd)

        # a token may not see the future, and a row may not see slots it has
        # not written. Both are "key index > this token's own position".
        k_pos = torch.arange(S, device=x.device)                          # [S]
        mask = k_pos > positions[:, :, None]                              # [rows, T, S]
        scores = scores.masked_fill(mask[:, None], float("-inf"))

        # softmax along the key axis, then weight the values
        out = scores.softmax(dim=-1) @ v_full            # [B, heads, T, head_dim]

        # concatenate heads, project back to the residual width
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, d_ff = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        # nn returns a copy of x, doesn't modify x in place
        y = self.SiLU(self.gate_proj(x))
        x = self.up_proj(x)
        # * doesn't mean dot product, it means element-wise
        x = x * y
        x = self.down_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache, layer_idx, rows, positions):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx,
                               rows, positions)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, input_ids, cache, rows: slice, positions):
        """input_ids and positions are both [rows, T]; rows selects cache rows."""
        x = self.embed_tokens(input_ids)
        cos, sin = rope_tables(self.cfg, positions)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, rows, positions)
        x = self.norm(x)
        # multiply by unembedding matrix which is embedding matrix but flipped
        return x @ self.embed_tokens.weight.T


"""
Exercise: add a KV cache to this working forward pass.

    uv run -m model.qwen_kv     # generates with and without the cache, diffs them

Four TODOs, marked below. The uncached path must keep working unchanged.
"""

import json
import math
from pathlib import Path

import torch
from analysis.reference import tokens
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

CONFIG = Path(__file__).resolve().parent / "configs" / "qwen3-4b.json"
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


def rope_tables(cfg: dict, seq_len: int, device=None, offset: int = 0):
    """cos, sin of shape [1, seq_len, head_dim], for positions [offset, offset+seq_len)."""
    hd, base = cfg["head_dim"], cfg["rope_theta"]
    # theta_k = base ** (-2k / hd), one per dimension pair
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    pos = torch.arange(offset, offset + seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)
    # Duplicated to full width so apply_rope is elementwise.
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos()[None], angles.sin()[None]


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
    """Post-RoPE keys and values, one entry per layer. Internal layout is up to you."""

    def __init__(self, n_layers: int):
        self.keys=[None] * n_layers
        self.values=[None] * n_layers

    def __len__(self) -> int:
        """Number of tokens currently cached."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[0]

    def append(self, layer_idx: int, k, v):
        """Store this step's k and v for one layer/block.

        k, v: [batch, T, n_kv_heads, head_dim], the new tokens only.
        Returns the full cached (K, V) for that layer, same layout.
        TODO: for now we're assuming batch is 1 because only serving 1 user at once
        keys:
        [ 
            [ block 0
                [ token 0
                    [ head 0
                        [128 dim], [128 dim], ... 8 of them]
                    ]
                ]
            ]
        ]
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = torch.tensor(k[0])
            self.values[layer_idx] = torch.tensor(v[0])
        else:
            # because of GQA a set of Q share one KV!
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], k[0]]) # torch.cat creates a new copy. In pytorch memory is contiguous and cannot 'append'
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v[0]])
        return (self.keys[layer_idx], self.values[layer_idx])

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

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        B, T, _ = x.shape
        group = self.n_heads // self.n_kv_heads          # 4 query heads share one kv head
        past = len(cache) if cache is not None else 0

        # project, then split the packed output into heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.hd)

        # per-head RMSNorm, then rotate by position
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        # if cache exists, append it!
        if len(cache) != 0:
            cache.append(layer_idx, k, v)

        S = k.shape[1]                                   # total keys: past + new

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

        # a token may not see the future. Query i sits at absolute position
        # past + i; key j sits at j. Forbidden when j > past + i.
        q_pos = torch.arange(past, past + T, device=x.device)[:, None]   # [T, 1]
        k_pos = torch.arange(S, device=x.device)[None, :]                # [1, S]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

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

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, input_ids, cache=None):
        x = self.embed_tokens(input_ids)
        past = len(cache) if cache is not None else 0
        cos, sin = rope_tables(self.cfg, input_ids.shape[1], x.device, offset=past)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i)
        x = self.norm(x)
        # multiply by unembedding matrix which is embedding matrix but flipped
        return x @ self.embed_tokens.weight.T


def generate(model, ids, max_new_tokens: int, use_cache: bool) -> list[int]:
    """Greedy decode. With a cache, only the newest token is fed after the first step."""
    cache = KVCache(model.cfg["num_hidden_layers"]) if use_cache else None
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            step = ids if cache is None or len(cache) == 0 else ids[:, -1:]
            next_id = model(step, cache)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            out.append(next_id.item())
    return out


def main():
    model = Qwen3(load_config())
    model.load_state_dict(load_weights(), strict=True)
    model = model.eval().to("mps", torch.float32)

    ids = tokens().to("mps")
    n = 20

    baseline = generate(model, ids, n, use_cache=False)
    cached = generate(model, ids, n, use_cache=True)

    print("no cache:", baseline)
    print("cached  :", cached)
    if baseline == cached:
        print("PASS")
    else:
        d = next(i for i, (a, b) in enumerate(zip(baseline, cached)) if a != b)
        print(f"FAIL -- diverges at token {d}")


if __name__ == "__main__":
    main()

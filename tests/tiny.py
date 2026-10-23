"""A Qwen3-shaped model small enough to run on a laptop CPU in milliseconds.

Same architecture as the real one -- QK-norm, RoPE, GQA, SwiGLU, tied embeddings
-- just narrow. Random weights, seeded, so a test failure reproduces.
"""

from types import SimpleNamespace

import torch

TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 100,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


def random_state(cls, cfg=TINY, seed=0, dtype=torch.float32):
    """A seeded state_dict with every parameter non-trivial (norm weights too)."""
    torch.manual_seed(seed)
    model = cls(cfg)
    sd = {}
    for name, p in model.state_dict().items():
        scale = 0.5 if p.dim() > 1 else 0.2
        sd[name] = (torch.randn_like(p) * scale + (1.0 if p.dim() == 1 else 0.0)).to(dtype)
    return sd


def build(cls, sd, cfg=TINY, dtype=torch.float32, device="cpu"):
    model = cls(cfg).to(dtype)
    model.load_state_dict(sd)
    return model.eval().to(device)


class StubTokenizer:
    """Just enough of the HF tokenizer surface that Engine.submit and record use."""

    all_special_ids = [0]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return messages[0]["content"]

    def __call__(self, text, return_tensors="pt"):
        ids = [1 + (ord(c) % (TINY["vocab_size"] - 1)) for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

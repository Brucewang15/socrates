"""Capture HuggingFace intermediates so you can diff your implementation piece by piece.

    uv run analysis/reference.py                 # capture -> model/reference.pt
    uv run analysis/reference.py --list          # show captured keys

In model/qwen/qwen.py (run from the repo root as `uv run -m model.qwen.qwen`):

    from analysis.reference import check, tokens
    check("layers.0.self_attn", my_attn_out)
"""

import argparse
from pathlib import Path

import torch

REF_PATH = Path(__file__).resolve().parent.parent / "model" / "reference.pt"
MODEL = "Qwen/Qwen3-4B"
PROMPT = "The capital of France is"

# Ordered to match the day-2 build order.
CAPTURE = [
    "model.embed_tokens",
    "model.rotary_emb",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.q_norm",
    "model.layers.0.self_attn.k_norm",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp",
    "model.layers.0",
    "model.layers.1",
    "model.norm",
]

_cache = None


def _store(name: str, out, into: dict) -> None:
    """Flatten a module output into tensors. Tuples with one tensor keep the bare name."""
    if torch.is_tensor(out):
        into[name] = out.detach().float().clone()
        return
    if isinstance(out, (tuple, list)):
        found = [o for o in out if torch.is_tensor(o)]
        if len(found) == 1:
            into[name] = found[0].detach().float().clone()
        else:
            for i, o in enumerate(out):
                if torch.is_tensor(o):
                    into[f"{name}.{i}"] = o.detach().float().clone()


def capture(model_id: str, device: str, dtype: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype)
    ).to(device).eval()

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    out: dict = {}
    handles = []
    by_name = dict(model.named_modules())
    for name in CAPTURE:
        if name not in by_name:
            print(f"  missing: {name}")
            continue
        handles.append(
            by_name[name].register_forward_hook(
                lambda m, i, o, n=name: _store(n, o, out)
            )
        )

    with torch.no_grad():
        logits = model(ids).logits
    for h in handles:
        h.remove()

    out["input_ids"] = ids.cpu()
    out["logits"] = logits.detach().float().cpu()
    return {k: v.cpu() for k, v in out.items()}


def load() -> dict:
    global _cache
    if _cache is None:
        if not REF_PATH.is_file():
            raise FileNotFoundError(f"{REF_PATH} missing -- run: uv run analysis/reference.py")
        _cache = torch.load(REF_PATH, map_location="cpu")
    return _cache


def tokens() -> torch.Tensor:
    """The input_ids the reference was captured with. Feed your model the same thing."""
    return load()["input_ids"]


def check(name: str, tensor: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Compare against the captured tensor. Accepts full or 'model.'-stripped names."""
    ref = load()
    key = name if name in ref else f"model.{name}"
    if key not in ref:
        raise KeyError(f"{name} not captured. Available: {sorted(ref)}")

    got = tensor.detach().float().cpu()
    exp = ref[key]
    if got.shape != exp.shape:
        print(f"FAIL {name}: shape {tuple(got.shape)} != {tuple(exp.shape)}")
        return False

    ok = torch.allclose(got, exp, rtol=rtol, atol=atol)
    diff = (got - exp).abs().max().item()
    print(f"{'ok  ' if ok else 'FAIL'} {name}  max_diff={diff:.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in sorted(load().items()):
            print(f"  {k:48s} {tuple(v.shape)}")
        return

    ref = capture(args.model, args.device, args.dtype)
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, REF_PATH)
    print(f"\n{len(ref)} tensors -> {REF_PATH}")
    for k, v in ref.items():
        print(f"  {k:48s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()

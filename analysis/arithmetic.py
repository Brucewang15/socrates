"""
Step 1: Derive a model's memory and speed limits from its config.json alone.

No GPU, no weight download -- just arithmetic. Everything you measure later in
this project gets compared against the numbers this script predicts.

Fill in the TODOs, then run:

    uv run arithmetic.py

The script checks your parameter count against the model's real one. When they
match, you understand the architecture completely.
"""

import argparse
import json

from huggingface_hub import get_safetensors_metadata, hf_hub_download

# Hardware profiles: (HBM capacity GB, memory bandwidth GB/s)
DEVICES = {
    "a10g": (24, 600),        # AWS g5.xlarge
    "l40s": (48, 864),        # AWS g6e.xlarge
    "m4pro": (48, 273),       # your laptop (unified memory)
}


def load_config(model_id: str) -> dict:
    path = hf_hub_download(model_id, filename="config.json")
    with open(path) as f:
        cfg = json.load(f)
    # Some multimodal repos nest the language model config one level down.
    return cfg.get("text_config", cfg)


def head_dim(cfg: dict) -> int:
    """Size of a single attention head."""
    if "head_dim" in cfg:
        return cfg["head_dim"]
    return cfg["hidden_size"] // cfg["num_attention_heads"]


def count_params(cfg: dict) -> dict[str, int]:
    """Return a dict of component name -> parameter count.

    Read these off the config: hidden_size (d), intermediate_size (d_ff),
    num_hidden_layers (N), num_attention_heads, num_key_value_heads, vocab_size.

    Remember a Linear(in, out) holds in*out weights. Qwen-style models have no
    biases on these projections, so you can ignore bias terms.
    """
    d = cfg["hidden_size"]
    d_ff = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    vocab = cfg["vocab_size"]
    hd = head_dim(cfg)

    # TODO(1): token embedding table -- one vector of width d per vocab entry.
    embeddings = 0

    # TODO(2): attention projections for ONE layer.
    #   q_proj: d -> n_heads * hd
    #   k_proj: d -> n_kv_heads * hd     <- smaller! this is GQA
    #   v_proj: d -> n_kv_heads * hd
    #   o_proj: n_heads * hd -> d
    attn_per_layer = 0

    # TODO(3): MLP projections for ONE layer (SwiGLU has THREE matrices).
    #   gate_proj: d -> d_ff
    #   up_proj:   d -> d_ff
    #   down_proj: d_ff -> d
    mlp_per_layer = 0

    # TODO(4): the output head, d -> vocab.
    # Careful: if cfg["tie_word_embeddings"] is True the model REUSES the
    # embedding table here and this costs zero extra parameters.
    lm_head = 0

    return {
        "embeddings": embeddings,
        "attention": attn_per_layer * n_layers,
        "mlp": mlp_per_layer * n_layers,
        "lm_head": lm_head,
    }


def kv_cache_bytes_per_token(cfg: dict, dtype_bytes: int = 2) -> int:
    """Bytes of KV cache one token occupies, across all layers.

    Every layer stores both a K and a V vector for each token, and each is
    (num_key_value_heads * head_dim) wide. Note this does NOT depend on
    num_attention_heads -- that's the whole point of GQA.
    """
    # TODO(5)
    return 0


def actual_param_count(model_id: str) -> int | None:
    """Ground truth, read from safetensors headers (no weight download)."""
    try:
        meta = get_safetensors_metadata(model_id)
        return sum(meta.parameter_count.values())
    except Exception as e:
        print(f"  (couldn't fetch ground truth: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="a10g", choices=DEVICES)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2=bf16, 1=fp8")
    args = ap.parse_args()

    cfg = load_config(args.model)
    vram_gb, bandwidth = DEVICES[args.device]

    print(f"\n{args.model}  on  {args.device}\n")
    print("  architecture")
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "vocab_size",
              "tie_word_embeddings"):
        if k in cfg:
            print(f"    {k:<24} {cfg[k]}")
    print(f"    {'head_dim':<24} {head_dim(cfg)}")

    parts = count_params(cfg)
    total = sum(parts.values())

    print("\n  parameters")
    for name, n in parts.items():
        share = f"{100 * n / total:5.1f}%" if total else "    -"
        print(f"    {name:<24} {n / 1e9:8.3f} B   {share}")
    print(f"    {'TOTAL':<24} {total / 1e9:8.3f} B")

    truth = actual_param_count(args.model)
    if truth:
        err = abs(total - truth) / truth * 100 if total else 100.0
        mark = "OK" if err < 1.0 else "MISMATCH"
        print(f"    {'actual':<24} {truth / 1e9:8.3f} B   <- {mark} ({err:.1f}% off)")

    if not total:
        print("\n  Fill in the TODOs above to see the memory analysis.\n")
        return

    # --- What the parameter count implies -------------------------------
    weight_gb = total * args.dtype_bytes / 1e9
    kv_per_tok = kv_cache_bytes_per_token(cfg, args.dtype_bytes)
    free_gb = vram_gb * 0.9 - weight_gb  # ~10% overhead for activations/fragmentation

    print("\n  memory")
    print(f"    weights                  {weight_gb:8.2f} GB")
    print(f"    free for KV cache        {free_gb:8.2f} GB  (of {vram_gb} GB, minus overhead)")

    if kv_per_tok:
        print(f"    KV cache per token       {kv_per_tok / 1024:8.1f} KB")
        if free_gb > 0:
            tokens = int(free_gb * 1e9 / kv_per_tok)
            print(f"    total cacheable tokens   {tokens:8,}")
            for seq in (1024, 4096):
                print(f"      -> at {seq:>5} ctx:        {tokens // seq:5,} concurrent requests")
    else:
        print("    KV cache per token          (TODO 5)")

    print("\n  speed ceiling (decode is memory-bandwidth bound)")
    print(f"    {bandwidth} GB/s / {weight_gb:.2f} GB per token = "
          f"{bandwidth / weight_gb:.1f} tok/s")
    print("    ^ this is your batch-size-1 upper bound. Measure it in step 2.\n")


if __name__ == "__main__":
    main()

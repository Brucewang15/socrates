"""Will torch.compile actually help the decode step, and at what granularity?

    uv run bench/day_5/compile_check.py

The win from mode="reduce-overhead" comes from CUDA graphs, and CUDA graphs need
Dynamo to trace the step into *one* graph. Whether it can is a tracing question,
not a device question -- so it is answerable on a laptop, on CPU, with random
weights, in seconds. What it cannot answer is speed: CUDA graphs do not exist on
CPU or MPS, so the payoff has to be measured on the GPU host.

Three granularities, because whole-model is not the only option:

    none        eager. the control -- does the engine still work at all
    full        torch.compile(model). one graph for the whole step if it traces
    per-block   torch.compile on each Block. 36 small graphs, compiled once and
                reused across layers

Whole-model gives the biggest CUDA graph and so the biggest cut to launch
overhead, but it recompiles for every distinct input shape and takes longest to
compile. Per-block compiles one artifact and reuses it 36 times, which is much
cheaper to build and more robust if something elsewhere breaks the trace -- at
the cost of 36 graph launches per step instead of 1.

Each mode is checked for graph count, break reasons, and whether it still emits
the same tokens as eager.
"""

import torch

from model.qwen.qwen_kv import KVCache as KVCache1
from model.qwen.qwen_kv import Qwen3 as Qwen3Batch1
from model.qwen.qwen_kv_cont import KVCache, Qwen3, rope_tables

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

ROWS = 2
PROMPT_LEN = 6
STEPS = 4


def build():
    torch.manual_seed(0)
    cfg = dict(TINY)
    model = Qwen3(cfg).eval().float()
    cache = KVCache(ROWS, cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                    cfg["head_dim"], max_len=64, dtype=torch.float32, device="cpu")
    # attached rather than passed to forward: that is what makes the cache module
    # state instead of a graph input, which is what CUDA graphs need
    model.attach_cache(cache)
    return cfg, model, cache


def prefill(model, cache, rows_n):
    """Fill every row, one at a time, exactly as Engine.prefill does."""
    out = []
    for row in range(rows_n):
        ids = torch.arange(PROMPT_LEN)[None] % TINY["vocab_size"] + row
        positions = torch.arange(ids.shape[1])[None]
        logits = model(ids, slice(row, row + 1), positions)
        cache.lengths[row] = ids.shape[1]        # caller owns lengths
        out.append([int(logits[:, -1].argmax(-1))])
    return out


def decode(model, cache, out, steps):
    """Batched decode, positions from each row's own cached length."""
    for _ in range(steps):
        ids = torch.tensor([[o[-1]] for o in out])
        positions = torch.tensor([[cache.lengths[r]] for r in range(len(out))])
        logits = model(ids, slice(0, len(out)), positions)
        for r in range(len(out)):
            cache.lengths[r] += 1                # caller owns lengths
        for i, tok in enumerate(logits[:, -1].argmax(-1).tolist()):
            out[i].append(int(tok))
    return out


def decode_args(model, cache, cfg, rows_n):
    """The exact tensors one decode step feeds the model."""
    ids = torch.tensor([[1]] * rows_n)
    positions = torch.tensor([[cache.lengths[r]] for r in range(rows_n)])
    return ids, positions


def explain(fn, *args):
    torch._dynamo.reset()
    e = torch._dynamo.explain(fn)(*args)
    reasons = []
    for r in e.break_reasons:
        line = str(getattr(r, "reason", r)).strip().splitlines()[0]
        if line not in reasons:
            reasons.append(line)
    return e.graph_count, e.graph_break_count, e.op_count, reasons


def reference_tokens(cfg, state, row):
    """Ground truth from the day-2 batch-1 path in model/qwen/qwen_kv.py.

    Independent of the ragged cache and of SDPA, so it catches a change in the
    attention math -- something comparing compiled against eager cannot do,
    since both would be wrong together.
    """
    m = Qwen3Batch1(cfg).eval().float()
    m.load_state_dict(state, strict=True)
    cache = KVCache1(cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                     cfg["head_dim"], max_len=64, dtype=torch.float32, device="cpu")
    ids = torch.arange(PROMPT_LEN)[None] % TINY["vocab_size"] + row
    out = []
    with torch.no_grad():
        step = ids
        for _ in range(STEPS + 1):
            tok = int(m(step, cache)[:, -1].argmax(-1))
            out.append(tok)
            step = torch.tensor([[tok]])
    return out


def main() -> None:
    print(f"torch {torch.__version__}, {TINY['num_hidden_layers']} layers, "
          f"{ROWS} rows\n")

    # ---- 0. does the ragged path still agree with the day-2 batch-1 path? ---
    cfg, model, cache = build()
    state = model.state_dict()
    want_ref = [reference_tokens(cfg, state, row) for row in range(ROWS)]
    got_ref = decode(model, cache, prefill(model, cache, ROWS), STEPS)
    ref_ok = got_ref == want_ref
    print(f"{'ok  ' if ref_ok else 'FAIL'} ragged+SDPA matches model/qwen/qwen_kv.py "
          f"(batch-1 reference)")
    if not ref_ok:
        for i, (w, g) in enumerate(zip(want_ref, got_ref)):
            if w != g:
                print(f"     row {i}: want {w}\n            got  {g}")
    print()

    # ---- what Dynamo makes of each granularity -----------------------------
    cfg, model, cache = build()
    prefill(model, cache, ROWS)
    ids, positions = decode_args(model, cache, cfg, ROWS)

    g_full, b_full, ops_full, why_full = explain(
        model, ids, slice(0, ROWS), positions)

    # one Block, given the tensors it would see mid-forward
    x = model.embed_tokens(ids)
    cos, sin = rope_tables(cfg, positions)
    g_blk, b_blk, ops_blk, why_blk = explain(
        model.layers[0], x, cos, sin, cache, 0, slice(0, ROWS), positions)

    print(f"{'granularity':<12} {'graphs':>7} {'breaks':>7} {'ops':>6}   notes")
    print(f"{'full model':<12} {g_full:>7} {b_full:>7} {ops_full:>6}   "
          f"{'one graph spans the step' if g_full == 1 and b_full == 0 else 'fragmented'}")
    per_step = g_blk * TINY["num_hidden_layers"]
    print(f"{'per-block':<12} {g_blk:>7} {b_blk:>7} {ops_blk:>6}   "
          f"{per_step} graph launches/step at {TINY['num_hidden_layers']} layers "
          f"({per_step * 18} at 36)")
    for label, why in (("full", why_full), ("block", why_blk)):
        for line in why:
            print(f"  [{label}] {line[:130]}")

    # ---- does each mode still produce the same tokens? ----------------------
    print()
    cfg, eager, cache_e = build()
    want = decode(eager, cache_e, prefill(eager, cache_e, ROWS), STEPS)
    print(f"ok   none       eager baseline, {len(want[0])} tokens/row")

    ok = True
    for mode in ("full", "per-block"):
        torch._dynamo.reset()
        cfg, m, c = build()
        if mode == "full":
            m_run = torch.compile(m, dynamic=False)
        else:
            for i, blk in enumerate(m.layers):
                m.layers[i] = torch.compile(blk, dynamic=False)
            m_run = m
        got = decode(m_run, c, prefill(m_run, c, ROWS), STEPS)
        same = got == want
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} {mode:<10} matches eager")
        if not same:
            for i, (w, g) in enumerate(zip(want, got)):
                if w != g:
                    print(f"       row {i}: want {w}\n              got  {g}")

    print("\nPASS" if (ok and ref_ok) else "\nFAIL")
    raise SystemExit(0 if (ok and ref_ok) else 1)


if __name__ == "__main__":
    main()

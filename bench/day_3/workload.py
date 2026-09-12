"""Shared workload, metrics and plots for the day-3 batching benchmarks.

Not a benchmark itself. Imported by static_batching.py and continuous_batching.py
so both policies see byte-identical requests -- otherwise the comparison means
nothing.

The two prompts are the ones this repo already uses:

    "how to make pizza?"        model/chat.py, bench/day_2/{no,with}_kv_cache.py
    "The capital of France is"  analysis/reference.py

Each is fed the way it is already fed: the pizza prompt through the chat
template with thinking disabled, the France prompt raw as a bare completion.
That difference is useful here -- it gives the batch two prompt lengths for
free, and ragged prompts are half the point of the exercise.

Request *lengths* are deliberately uneven, which is the whole reason continuous
batching exists. The default shape follows the day-4 sketch in the README: a
few long requests among many short ones, so under static batching the long one
holds a mostly-idle wave open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from model.qwen_batch import Request

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Prompt, and where it is already used in the repo.
CHAT_PROMPT = "how to make pizza?"              # model/chat.py, bench/day_2/*
RAW_PROMPT = "The capital of France is"         # analysis/reference.py


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags shared by both benchmarks, so the workloads stay comparable."""
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=8,
                    help="rows in the KV cache, i.e. the concurrency ceiling")
    ap.add_argument("--short-tokens", type=int, default=24)
    ap.add_argument("--long-tokens", type=int, default=160)
    ap.add_argument("--long-every", type=int, default=8,
                    help="every Nth request is a long one")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--dtype", default="bfloat16")
    return ap


def prompt_text(tok, which: str) -> str:
    """Render a prompt exactly as the existing scripts render it."""
    if which == "chat":
        return tok.apply_chat_template(
            [{"role": "user", "content": CHAT_PROMPT}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    return RAW_PROMPT


def build_requests(tok, args, device) -> list[Request]:
    """Alternate the two prompts; sprinkle long requests among short ones."""
    requests = []
    for i in range(args.requests):
        which = "chat" if i % 2 == 0 else "raw"
        text = prompt_text(tok, which)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        is_long = (i % args.long_every) == 0
        requests.append(Request(
            rid=i,
            input_ids=ids,
            max_new_tokens=args.long_tokens if is_long else args.short_tokens,
            prompt=CHAT_PROMPT if which == "chat" else RAW_PROMPT,
            label=f"{which}/{'long' if is_long else 'short'}",
        ))
    return requests


def required_max_len(requests: list[Request]) -> int:
    """Longest row any request could need. Sizing the cache is day-1 arithmetic."""
    return max(r.prompt_len + r.max_new_tokens for r in requests)


def print_workload(requests: list[Request], max_batch: int, max_len: int) -> None:
    total = sum(r.max_new_tokens for r in requests)
    print(f"{len(requests)} requests | {total} tokens requested | "
          f"max_batch={max_batch} | max_len={max_len}")
    for r in requests:
        print(f"  r{r.rid:<3d} {r.label:<12s} prompt={r.prompt_len:<4d} "
              f"want={r.max_new_tokens}")


def pct(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0


def metrics(result) -> dict:
    """Everything the writeup table wants, in one dict."""
    lat = [r.latency for r in result.requests]
    ttft = [r.ttft for r in result.requests]
    queued = [r.queued for r in result.requests]
    decode_ms = [ms for ms, k in zip(result.step_ms, result.step_kind) if k == "decode"]
    return {
        "policy": result.policy,
        "requests": len(result.requests),
        "out_tokens": result.out_tokens,
        "wall_s": result.wall,
        "throughput_tok_s": result.throughput,
        "forwards": len(result.step_ms),
        "decode_steps": len(decode_ms),
        "mean_decode_batch": result.mean_batch,
        "decode_ms_p50": pct(decode_ms, 50),
        "ttft_p50": pct(ttft, 50),
        "ttft_p99": pct(ttft, 99),
        "latency_p50": pct(lat, 50),
        "latency_p99": pct(lat, 99),
        "queued_p50": pct(queued, 50),
        "queued_max": max(queued) if queued else 0.0,
        "cache_bytes": result.cache_bytes,
    }


def print_metrics(m: dict) -> None:
    print(f"\n{m['policy']} batching")
    print(f"  wall                {m['wall_s']:.2f} s")
    print(f"  output tokens       {m['out_tokens']}")
    print(f"  throughput          {m['throughput_tok_s']:.2f} tok/s")
    print(f"  forwards            {m['forwards']} ({m['decode_steps']} decode)")
    print(f"  mean decode batch   {m['mean_decode_batch']:.2f} of {m['requests']} requests")
    print(f"  decode step p50     {m['decode_ms_p50']:.1f} ms")
    print(f"  TTFT   p50 / p99    {m['ttft_p50']*1000:.0f} / {m['ttft_p99']*1000:.0f} ms")
    print(f"  latency p50 / p99   {m['latency_p50']:.2f} / {m['latency_p99']:.2f} s")
    print(f"  queued  p50 / max   {m['queued_p50']:.2f} / {m['queued_max']:.2f} s")
    print(f"  cache reserved      {m['cache_bytes'] / 1e9:.2f} GB")


def save(result, stem: str) -> Path:
    """One npz per policy so compare.py can plot both without re-running."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stem}.npz"
    reqs = result.requests
    t0 = reqs[0].arrival
    np.savez(
        path,
        meta=json.dumps(metrics(result)),
        policy=result.policy,
        step_t=np.array(result.step_t),
        step_batch=np.array(result.step_batch),
        step_ms=np.array(result.step_ms),
        step_is_decode=np.array([k == "decode" for k in result.step_kind]),
        rid=np.array([r.rid for r in reqs]),
        prompt_len=np.array([r.prompt_len for r in reqs]),
        generated=np.array([r.n_generated for r in reqs]),
        want=np.array([r.max_new_tokens for r in reqs]),
        admit_rel=np.array([r.t_admit - t0 for r in reqs]),
        first_rel=np.array([r.t_first_token - t0 for r in reqs]),
        done_rel=np.array([r.t_done - t0 for r in reqs]),
        label=np.array([r.label for r in reqs]),
    )
    print(f"saved {path}")
    return path


def plot_run(result, stem: str, color: str, subtitle: str) -> Path:
    """Occupancy over time, and a per-request timeline showing queue wait."""
    import matplotlib

    matplotlib.use("Agg")            # file output only, no GUI backend
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{stem}.png"
    m = metrics(result)
    reqs = result.requests
    t0 = reqs[0].arrival

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    t = np.array(result.step_t)
    b = np.array(result.step_batch)
    ax1.step(t, b, where="post", lw=1.4, color=color)
    ax1.axhline(m["mean_decode_batch"], ls="--", lw=1, color="#dc2626",
                label=f"mean decode batch {m['mean_decode_batch']:.2f}")
    ax1.set(xlabel="time (s)", ylabel="rows live",
            title="batch occupancy", ylim=(0, None))
    ax1.legend(fontsize=8, loc="lower left")

    for r in reqs:
        y = r.rid
        admit, first, done = r.t_admit - t0, r.t_first_token - t0, r.t_done - t0
        if admit > 0:
            ax2.plot([0, admit], [y, y], lw=3, color="#cbd5e1",
                     solid_capstyle="butt")          # queued, not yet admitted
        ax2.plot([admit, done], [y, y], lw=3, color=color, solid_capstyle="butt")
        ax2.plot([first], [y], "|", ms=7, color="#0f172a")
    ax2.set(xlabel="time (s)", ylabel="request", title="per-request timeline")
    ax2.invert_yaxis()

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{result.policy} batching -- {subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    return out


def load_model(args):
    """Load Qwen3-4B into the ragged-batch model. Shared so both runs match."""
    from model.qwen_batch import Qwen3Batch, load_config, load_weights, pick_device
    from transformers import AutoTokenizer

    from model.qwen_batch import MODEL_ID

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = load_config()
    model = Qwen3Batch(cfg)
    model.load_state_dict(load_weights(), strict=True)
    model = model.eval().to(device, dtype)
    return tok, cfg, model, device

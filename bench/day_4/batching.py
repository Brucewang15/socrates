"""Static vs continuous batching: runs both engines, writes the numbers.

    uv run bench/day_4/batching.py                       # both, sequentially
    uv run bench/day_4/batching.py --policy continuous
    uv run bench/day_4/batching.py --max-batch 8 --max-new-tokens 300

model/inference_static.py pads a wave and marches it in lockstep;
model/inference_cont.py refills a freed row the same step. This file submits
the same prompts to each, times them, and derives every metric from the
timestamps each Request already carries. It plots nothing -- run plots.py.

Writes bench/results/day4_batching.npz.
"""

import argparse
import gc
import time
from pathlib import Path

import model.inference_cont as cont
import model.inference_static as static
import numpy as np
import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"
STEM = "day4_batching"
GRID = 400

JOBS = [
    "how to make pizza?",
    "what is 2+2?",
    "who are you?",
    "name a color",
    "where is mao ze dong born",
    "who runs China?",
    "what do you think of the CCP?",
    "what happened in tianmen square 1989?",
    "explain recursion",
    "what is 5*5?",
    "name a fruit",
    "capital of France?",
    "how does a KV cache work?",
    "name an animal",
    "what is 10-7?",
    "write a haiku about GPUs",
]


WARMUP = ["say hi", "what is 2+2?", "name a color", "capital of France?"]


def warm(engine, policy, args) -> None:
    """Burn the one-time costs before the clock starts.

    The continuous engine compiles its decode step on first use -- tens of
    seconds, and again for each distinct row count as rows drain. The static
    engine does not compile at all, so timing that compilation inside the run
    both understates continuous throughput and hands static an unearned win.
    cuBLAS autotuning and lazy CUDA init are smaller versions of the same thing,
    which is why static gets a warmup too.

    Enough prompts to reach max_batch, so every row count that the real run will
    trace has already been traced.
    """
    prompts = (WARMUP * 8)[:max(args.max_batch, 2)]
    if policy == "static":
        engine.run_batch([static.Request(p) for p in prompts])
    else:
        for p in prompts:
            engine.submit(p)
        engine.run()


def run_static(args):
    static.BATCH_SIZE = args.max_batch
    static.MAX_NEW_TOKENS = args.max_new_tokens
    static.DEVICE = args.device
    engine = static.Engine()
    warm(engine, "static", args)

    reqs = [static.Request(p) for p in JOBS]
    t0 = time.perf_counter()
    for r in reqs:
        r.submitted = t0
    # Engine.loop() never returns, so cut the waves on its own rule
    for i in range(0, len(reqs), args.max_batch):
        engine.run_batch(reqs[i:i + args.max_batch])
    return engine, reqs, time.perf_counter() - t0


def run_continuous(args):
    cont.MAX_BATCH = args.max_batch
    cont.MAX_NEW_TOKENS = args.max_new_tokens
    cont.DEVICE = args.device
    engine = cont.Engine()
    warm(engine, "continuous", args)

    # static encodes inside run_batch, after its t0; start the clock before
    # submit() so both pay tokenization inside the timed region
    t0 = time.perf_counter()
    reqs = [engine.submit(p) for p in JOBS]
    for r in reqs:
        r.submitted = t0
    engine.run()
    return engine, reqs, time.perf_counter() - t0


def prompt_lengths(tok) -> np.ndarray:
    """Prompt tokens per job, rendered the way both engines render them."""
    out = []
    for p in JOBS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        out.append(tok(text, return_tensors="pt").input_ids.shape[1])
    return np.array(out)


def occupancy(start, end, wall):
    """Rows busy over time, from each request's own span."""
    t = np.linspace(0, wall, GRID)
    return t, ((start[:, None] <= t) & (t <= end[:, None])).sum(0)


def cache_bytes(cfg, rows, max_len, dtype) -> int:
    itemsize = torch.empty((), dtype=dtype).element_size()
    per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] \
        * cfg["head_dim"] * itemsize
    return per_token * rows * max_len


def score(reqs, wall, tok, cfg, args, policy) -> dict:
    tokens = np.array([len(r.output) for r in reqs])
    response = np.array([r.first_token - r.submitted for r in reqs])
    own = np.array([r.finished - r.submitted for r in reqs])
    # only static has `delivered`; a continuous job is handed back when it ends
    delivered = np.array([getattr(r, "delivered", r.finished) - r.submitted
                          for r in reqs])
    plen = prompt_lengths(tok)

    # static pads each wave to its longest prompt; ragged rows never pad
    pad = 0
    if policy == "static":
        for i in range(0, len(plen), args.max_batch):
            w = plen[i:i + args.max_batch]
            pad += int((w.max() - w).sum())

    computed = sum(getattr(r, "steps", 0) for r in reqs)
    # A row is occupied from admission, not from its first token -- prefill holds
    # it too. cont sets `admitted`; static admits a whole wave, so its first
    # token is the closest marker it has.
    admitted = np.array([getattr(r, "admitted", r.first_token) - r.submitted
                         for r in reqs])
    t, live = occupancy(response, own, wall)
    _, held = occupancy(admitted, delivered, wall)

    return {
        "tokens": tokens, "response": response, "own": own, "delivered": delivered,
        "idle": delivered - own, "prompt_len": plen,
        "occ_t": t, "occ_live": live, "occ_held": held,
        "wall": wall,
        "out_tokens": tokens.sum(),
        "throughput": tokens.sum() / wall,
        "mean_live": float(live.mean()),
        "mean_held": float(held.mean()),
        "ttft_p50": float(np.percentile(response, 50)),
        "ttft_p99": float(np.percentile(response, 99)),
        "latency_p50": float(np.percentile(own, 50)),
        "latency_p99": float(np.percentile(own, 99)),
        "released_max": float(delivered.max()),
        "idle_total": float((delivered - own).sum()),
        "padded_tokens": pad,
        "goodput": tokens.sum() / computed if computed else float("nan"),
        "cache_bytes": cache_bytes(cfg, args.max_batch, 2048, torch.bfloat16),
    }


def report(policy, reqs, m, tok, max_batch) -> None:
    print(f"\n{policy} batching")
    print(f"  {'job':<34}{'tok':>5}{'response':>10}{'done':>8}"
          f"{'released':>10}{'idle':>8}")
    for r, n, resp, own, rel, idle in zip(reqs, m["tokens"], m["response"],
                                          m["own"], m["delivered"], m["idle"]):
        print(f"  {r.prompt[:34]:<34}{n:>5}{resp:>9.2f}s{own:>7.2f}s"
              f"{rel:>9.2f}s{idle:>7.2f}s")
    print()
    print(f"  wall                {m['wall']:.2f} s")
    print(f"  output tokens       {m['out_tokens']}")
    print(f"  throughput          {m['throughput']:.2f} tok/s")
    print(f"  mean rows live      {m['mean_live']:.2f} of {max_batch}")
    print(f"  mean rows held      {m['mean_held']:.2f} of {max_batch}")
    print(f"  TTFT   p50 / p99    {m['ttft_p50']:.2f} / {m['ttft_p99']:.2f} s")
    print(f"  latency p50 / p99   {m['latency_p50']:.2f} / {m['latency_p99']:.2f} s")
    print(f"  last job released   {m['released_max']:.2f} s")
    print(f"  time jobs sat idle  {m['idle_total']:.1f} s")
    print(f"  pad tokens prefilled {m['padded_tokens']}")
    if not np.isnan(m["goodput"]):
        print(f"  goodput             {m['goodput']:.0%} of computed rows "
              f"produced a token")
    print(f"  cache reserved      {m['cache_bytes'] / 1e9:.2f} GB")
    print(f"  first job: {tok.decode(reqs[0].output)[:120]!r}")


ROWS = [
    ("wall (s)", "wall", "{:.2f}", "lower"),
    ("throughput (tok/s)", "throughput", "{:.2f}", "higher"),
    ("mean rows live", "mean_live", "{:.2f}", "higher"),
    ("TTFT p50 (s)", "ttft_p50", "{:.2f}", "lower"),
    ("TTFT p99 (s)", "ttft_p99", "{:.2f}", "lower"),
    ("latency p50 (s)", "latency_p50", "{:.2f}", "lower"),
    ("latency p99 (s)", "latency_p99", "{:.2f}", "lower"),
    ("last job released (s)", "released_max", "{:.2f}", "lower"),
    ("idle total (s)", "idle_total", "{:.1f}", "lower"),
]


def compare(runs) -> None:
    s, c = runs["static"], runs["continuous"]
    w = max(len(label) for label, *_ in ROWS)
    print(f"\n{'':<{w}}  {'static':>10}  {'continuous':>10}  {'ratio':>8}")
    for label, key, fmt, better in ROWS:
        a, b = float(s[key]), float(c[key])
        hi, lo = (b, a) if better == "higher" else (a, b)
        if lo > 0:
            note = f"{hi / lo:>6.2f}x {'better' if hi > lo else 'worse'}"
        else:
            note = f"{'--':>6}  {'eliminated' if hi > 0 else 'both zero'}"
        print(f"{label:<{w}}  {fmt.format(a):>10}  {fmt.format(b):>10}  {note}")
    print(f"\nthroughput {c['throughput'] / s['throughput']:.2f}x on "
          f"{c['mean_live'] / s['mean_live']:.2f}x the mean rows live.")
    print("They should track each other; if throughput lags, decode is drifting")
    print("compute-bound. Static also pads and attends to its pad tokens, so")
    print("greedy output can diverge -- treat the ratio as approximate.")


def free(device) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="both",
                    choices=["both", "static", "continuous"])
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="mps", help="mps, cuda or cpu")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    runners = {"static": run_static, "continuous": run_continuous}
    policies = list(runners) if args.policy == "both" else [args.policy]

    runs = {}
    for policy in policies:
        print(f"\nloading {policy} engine ({args.max_batch} rows, "
              f"{args.max_new_tokens} max new tokens)")
        engine, reqs, wall = runners[policy](args)
        runs[policy] = m = score(reqs, wall, engine.tok, engine.cfg, args, policy)
        report(policy, reqs, m, engine.tok, args.max_batch)
        # drop this engine before building the next: each model is ~8 GB
        del engine, reqs
        free(args.device)

    if len(runs) == 2:
        compare(runs)

    out = RESULTS / f"{STEM}.npz"
    np.savez(out, jobs=np.array(JOBS), policies=np.array(list(runs)),
             max_batch=args.max_batch, device=args.device,
             **{f"{p}_{k}": v for p, m in runs.items() for k, v in m.items()})
    print(f"\nsaved {out}\n  now run: uv run bench/day_4/plots.py")


if __name__ == "__main__":
    main()

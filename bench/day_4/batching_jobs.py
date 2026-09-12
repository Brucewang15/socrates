"""The 8-job timeline from bench/results/batching.png, under both policies.

    uv run bench/day_4/batching_jobs.py                  # both, batch of 4
    uv run bench/day_4/batching_jobs.py --policy continuous
    uv run bench/day_4/batching_jobs.py --max-batch 4 --max-new-tokens 300

Same eight prompts, all submitted at t=0, same bars as the original chart:

    red dot   response time      when the caller sees the first token
    purple    turnaround time    submission -> this job's last token
    grey      waiting            job is finished but not yet handed back

That grey band is the indictment of static batching. A wave is only released
when its slowest member finishes, so "what is 2+2?" sits completed and idle for
tens of seconds. Under continuous batching a job is returned the moment it
ends, so the grey should vanish -- and the last four jobs should start far
earlier, because they board a free seat instead of waiting for a whole wave.

Writes bench/results/batching_{policy}.png. The original batching.png is left
alone.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from static_batching import run_static
from workload import RESULTS, load_model, load_seq_model

from backend.inference_cont import Engine, Request

# Exactly the jobs in bench/day_3/batching.py, in the same order. The original
# chart truncates its y-labels to 30 characters, which is why the last one reads
# "what happened in tianmen squar" there.
JOBS = [
    "how to make pizza?",
    "what is 2+2?",
    "who are you?",
    "name a color",
    "where is mao ze dong born",
    "who runs China?",
    "what do you think of the CCP?",
    "what happened in tianmen square 1989?",
]

PURPLE = "#8a2be2"
GREY = "#e5e7eb"
RED = "#dc2626"


def build(tok, device, max_new_tokens) -> list[Request]:
    """One request per job. Chat template with thinking off, as model/chat.py does."""
    requests = []
    for i, prompt in enumerate(JOBS):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        requests.append(Request(rid=i, input_ids=ids, prompt=prompt,
                                max_new_tokens=max_new_tokens, label=prompt))
    return requests


def release_times(result) -> list[float]:
    """When each job is handed back to its caller, relative to submission.

    Continuous batching returns a job the instant it finishes. Static batching
    holds the whole wave: everyone admitted together leaves together, so the
    release time is the wave's slowest finisher. Waves are identified by shared
    admission timestamps, which is exactly how Engine.admit stamps them.
    """
    t0 = result.requests[0].arrival
    if result.policy == "continuous":
        return [r.t_done - t0 for r in result.requests]

    waves: dict[float, float] = {}
    for r in result.requests:
        waves[r.t_admit] = max(waves.get(r.t_admit, 0.0), r.t_done)
    return [waves[r.t_admit] - t0 for r in result.requests]


def plot(result, released, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")            # file output only, no GUI backend
    import matplotlib.pyplot as plt

    reqs = result.requests
    t0 = reqs[0].arrival
    turnaround = [r.t_done - t0 for r in reqs]
    first = [r.t_first_token - t0 for r in reqs]
    y = np.arange(len(reqs))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.barh(y, turnaround, height=0.72, color=PURPLE, label="turnaround time",
            zorder=3)
    ax.barh(y, [rel - end for rel, end in zip(released, turnaround)],
            left=turnaround, height=0.72, color=GREY,
            label="waiting for the batch", zorder=2)
    ax.plot(first, y, "o", ms=7, color=RED, label="response time", zorder=4)

    ax.set_yticks(y, [r.prompt[:30] for r in reqs])   # truncated, as the original does
    ax.set_xlabel("seconds since submission")
    ax.set_title(f"{result.policy.capitalize()} batching, {len(reqs)} jobs, "
                 f"batches of {args.max_batch}")
    ax.set_xlim(0, max(released) * 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("response time"), labels.index("waiting for the batch"),
             labels.index("turnaround time")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def report(result, released) -> dict:
    t0 = result.requests[0].arrival
    turnaround = [r.t_done - t0 for r in result.requests]
    waiting = [rel - end for rel, end in zip(released, turnaround)]
    ttft = [r.t_first_token - t0 for r in result.requests]

    print(f"\n{result.policy} batching, batch of {result.step_batch[0]} seats")
    print(f"  {'job':<32} {'first tok':>10} {'done':>8} {'released':>9} "
          f"{'idle':>7} {'tokens':>7}")
    for r, f, end, rel, w in zip(result.requests, ttft, turnaround, released, waiting):
        print(f"  {r.prompt:<32} {f:>9.1f}s {end:>7.1f}s {rel:>8.1f}s "
              f"{w:>6.1f}s {r.n_generated:>7}")
    print(f"  wall {result.wall:.1f}s | {result.out_tokens} tokens | "
          f"{result.throughput:.2f} tok/s | mean decode batch "
          f"{result.mean_batch:.2f} | idle total {sum(waiting):.1f}s")

    return {
        "policy": result.policy,
        "wall_s": result.wall,
        "out_tokens": result.out_tokens,
        "throughput_tok_s": result.throughput,
        "mean_decode_batch": result.mean_batch,
        "last_release_s": max(released),
        "idle_total_s": float(sum(waiting)),
        "ttft_mean_s": float(np.mean(ttft)),
        "turnaround_mean_s": float(np.mean(turnaround)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="both",
                    choices=["both", "continuous", "static"])
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="cap; short jobs stop earlier on EOS. "
                         "512 matches bench/day_3/batching.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    tok, cfg, model, device = (None,) * 4
    policies = ["static", "continuous"] if args.policy == "both" else [args.policy]

    summaries = {}
    for policy in policies:
        # one model at a time: the two are 8 GB each, and 16 GB resident would
        # swap on a 32 GB machine
        if policy == "static":
            tok, cfg, model, device = load_seq_model(args)
            requests = build(tok, device, args.max_new_tokens)
            print(f"\nrunning static: {len(requests)} jobs, {args.max_batch} seats")
            result = run_static(model, cfg, tok, requests, max_batch=args.max_batch,
                                device=device, dtype=getattr(torch, args.dtype))
        else:
            tok, cfg, model, device = load_model(args)
            requests = build(tok, device, args.max_new_tokens)
            max_len = max(r.prompt_len + r.max_new_tokens for r in requests)
            engine = Engine(model, cfg, stop_ids=tok.all_special_ids,
                            max_batch=args.max_batch, max_len=max_len)
            print(f"\nrunning continuous: {len(requests)} jobs, {args.max_batch} seats, "
                  f"cache {engine.cache.bytes_reserved / 1e9:.2f} GB")
            result = engine.run(requests, progress=True)

        released = release_times(result)
        summaries[policy] = report(result, released)

        RESULTS.mkdir(parents=True, exist_ok=True)
        plot(result, released, args, RESULTS / f"batching_{policy}.png")

        del model, result
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if len(summaries) == 2:
        s, c = summaries["static"], summaries["continuous"]
        print("\nstatic -> continuous")
        print(f"  last job released   {s['last_release_s']:.1f}s -> "
              f"{c['last_release_s']:.1f}s  ({s['last_release_s'] / c['last_release_s']:.2f}x)")
        print(f"  mean turnaround     {s['turnaround_mean_s']:.1f}s -> "
              f"{c['turnaround_mean_s']:.1f}s")
        print(f"  mean response time  {s['ttft_mean_s']:.1f}s -> {c['ttft_mean_s']:.1f}s")
        print(f"  time jobs sat idle  {s['idle_total_s']:.1f}s -> {c['idle_total_s']:.1f}s")
        print(f"  throughput          {s['throughput_tok_s']:.2f} -> "
              f"{c['throughput_tok_s']:.2f} tok/s")


if __name__ == "__main__":
    main()

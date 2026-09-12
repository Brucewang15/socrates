"""Static vs continuous batching, side by side. Reads the saved npz files.

    uv run bench/day_3/static_batching.py
    uv run bench/day_3/continuous_batching.py
    uv run bench/day_3/compare.py

Run both benchmarks first with the same flags -- compare.py refuses to compare
runs that did different amounts of work, since a throughput ratio between
different workloads is meaningless.

Prints the writeup row (static, continuous, ratio) and saves an overlay plot to
bench/results/batching_compare.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "batching_compare.png"

ROWS = [
    ("wall (s)", "wall_s", "{:.2f}", "lower"),
    ("throughput (tok/s)", "throughput_tok_s", "{:.2f}", "higher"),
    ("decode steps", "decode_steps", "{:.0f}", "lower"),
    ("mean decode batch", "mean_decode_batch", "{:.2f}", "higher"),
    ("decode step p50 (ms)", "decode_ms_p50", "{:.1f}", "lower"),
    ("TTFT p50 (ms)", "ttft_p50", "{:.0f}", "lower"),
    ("TTFT p99 (ms)", "ttft_p99", "{:.0f}", "lower"),
    ("latency p50 (s)", "latency_p50", "{:.2f}", "lower"),
    ("latency p99 (s)", "latency_p99", "{:.2f}", "lower"),
    ("queued max (s)", "queued_max", "{:.2f}", "lower"),
]


def load(stem: str):
    path = RESULTS / f"{stem}.npz"
    if not path.is_file():
        raise SystemExit(f"{path} missing -- run: uv run bench/day_3/{stem}.py")
    z = np.load(path, allow_pickle=False)
    return z, json.loads(str(z["meta"]))


def scale(key: str, value: float) -> float:
    """Report the two time-to-first-token rows in ms, everything else as stored."""
    return value * 1000 if key.startswith("ttft") else value


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    zs, ms = load("static_batching")
    zc, mc = load("continuous_batching")

    if ms["out_tokens"] != mc["out_tokens"] or ms["requests"] != mc["requests"]:
        raise SystemExit(
            f"workloads differ: static produced {ms['out_tokens']} tokens over "
            f"{ms['requests']} requests, continuous {mc['out_tokens']} over "
            f"{mc['requests']}. Re-run both with the same flags."
        )

    width = max(len(label) for label, *_ in ROWS)
    print(f"{mc['out_tokens']} output tokens, {mc['requests']} requests, "
          f"identical workload\n")
    print(f"{'':<{width}}  {'static':>10}  {'continuous':>10}  {'ratio':>8}")
    for label, key, fmt, better in ROWS:
        s, c = scale(key, ms[key]), scale(key, mc[key])
        ratio = (c / s if better == "higher" else s / c) if s and c else float("nan")
        verdict = "better" if ratio > 1 else "worse"
        print(f"{label:<{width}}  {fmt.format(s):>10}  {fmt.format(c):>10}  "
              f"{ratio:>6.2f}x {verdict}")

    speedup = mc["throughput_tok_s"] / ms["throughput_tok_s"]
    batch_ratio = mc["mean_decode_batch"] / ms["mean_decode_batch"]
    print(f"\nthroughput {speedup:.2f}x on {batch_ratio:.2f}x the mean batch size")
    print("The two should track each other. If throughput lags the batch ratio,")
    print("the extra rows are no longer free -- decode is drifting compute-bound.")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

    for z, m, color in ((zs, ms, "#2563eb"), (zc, mc, "#16a34a")):
        ax1.step(z["step_t"], z["step_batch"], where="post", lw=1.4,
                 color=color, label=f"{m['policy']} (mean {m['mean_decode_batch']:.1f})")
    ax1.set(xlabel="time (s)", ylabel="rows live", title="batch occupancy",
            ylim=(0, None))
    ax1.legend(fontsize=8, loc="lower left")

    order = np.argsort(zc["want"])
    idx = np.arange(len(order))
    ax2.barh(idx - 0.2, zs["done_rel"][order], height=0.4, color="#2563eb",
             label="static")
    ax2.barh(idx + 0.2, zc["done_rel"][order], height=0.4, color="#16a34a",
             label="continuous")
    ax2.set(xlabel="finish time (s)", ylabel="request, sorted by tokens wanted",
            title="when each request finished")
    ax2.set_yticks(idx[::2], [str(w) for w in zc["want"][order][::2]], fontsize=7)
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    labels = ["throughput\n(tok/s)", "TTFT p99\n(ms)", "latency p50\n(s)"]
    fmts = ["{:.0f}", "{:.0f}", "{:.2f}"]
    static_vals = [ms["throughput_tok_s"], ms["ttft_p99"] * 1000, ms["latency_p50"]]
    cont_vals = [mc["throughput_tok_s"], mc["ttft_p99"] * 1000, mc["latency_p50"]]
    x = np.arange(len(labels))
    # each metric has its own units, so normalise to the static bar and print
    # the real value on top of each one
    ax3.bar(x - 0.2, [1] * len(labels), width=0.4, color="#2563eb", label="static")
    ax3.bar(x + 0.2, [c / s if s else 0 for c, s in zip(cont_vals, static_vals)],
            width=0.4, color="#16a34a", label="continuous")
    for i, (s, c, fmt) in enumerate(zip(static_vals, cont_vals, fmts)):
        ax3.text(i - 0.2, 1.02, fmt.format(s), ha="center", fontsize=7)
        ax3.text(i + 0.2, (c / s if s else 0) + 0.02, fmt.format(c),
                 ha="center", fontsize=7)
    ax3.axhline(1, lw=0.8, color="#334155")
    ax3.set(xticks=x, ylabel="relative to static", title="headline numbers")
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.legend(fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("static vs continuous batching, identical workload and kernels",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()

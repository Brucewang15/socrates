"""Per-job response and turnaround time for one static batch.

    uv run bench/day_3/batching.py
"""

import time
from pathlib import Path

import backend.inference as inf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

BATCH_SIZE = 4
NEW_TOKENS = 512
PROMPTS = [
    "how to make pizza?",
    "what is 2+2?",
    "who are you?",
    "name a color",
    "where is mao ze dong born",
    "who runs China?",
    "what do you think of the CCP?",
    "what happened in tianmen square 1989?",
]
OUT = Path(__file__).resolve().parents[1] / "results" / "batching.png"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    inf.MAX_NEW_TOKENS = NEW_TOKENS
    engine = inf.Engine()

    reqs = [inf.Request(p) for p in PROMPTS]
    for req in reqs:
        engine.submit(req)
    while engine.pending.qsize() >= BATCH_SIZE:
        engine.run_batch([engine.pending.get() for _ in range(BATCH_SIZE)])

    response = np.array([r.first_token - r.submitted for r in reqs])
    own = np.array([r.finished - r.submitted for r in reqs])
    turnaround = np.array([r.delivered - r.submitted for r in reqs])
    tokens = np.array([len(r.output) for r in reqs])
    wall = turnaround.max()
    computed = sum(r.steps for r in reqs)

    print(f"{'prompt':40s} {'tok':>5s} {'response':>9s} {'own':>7s} {'delivered':>10s}")
    for r, resp, o, t, n in zip(reqs, response, own, turnaround, tokens):
        print(f"{r.prompt[:40]:40s} {n:5d} {resp:8.2f}s {o:6.2f}s {t:9.2f}s")
    print(f"\nwall {wall:.2f}s | {tokens.sum()} tok | {tokens.sum() / wall:.2f} tok/s "
          f"| goodput {tokens.sum() / computed:.0%}")

    np.savez(OUT.with_suffix(".npz"), response=response, own=own,
             turnaround=turnaround, tokens=tokens, wall=wall)

    y = np.arange(len(reqs))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y, turnaround, color="#e5e7eb", label="idle compute")
    ax.barh(y, own, color="#7c3aed", label="turnaround time")
    ax.plot(response, y, "o", color="#dc2626", ms=5, label="response time")
    ax.set_yticks(y, [p[:30] for p in PROMPTS], fontsize=8)
    ax.set_xlabel("seconds since submission")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Static batching, {len(PROMPTS)} jobs, batches of {BATCH_SIZE}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

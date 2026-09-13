"""Where the GPU time goes on the A10G. Reads the reduced trace summary.

    uv run bench/day_5/plot_gpu_profile.py

Input is bench/results/gpu_profile_summary.json, produced by
infra/prod/analyze_trace.py from a torch.profiler Chrome trace captured on the
deployed GPU host. The raw trace is 114 MB and stays on the box; this is the
few-KB reduction of it.

Four panels:

    occupancy   GPU-busy fraction over time, against host launch activity
    budget      the profiled window split into GPU-busy and GPU-idle
    ops         top CPU ops by self time, split overhead vs real compute
    kernels     where the little GPU time there is actually goes
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # file output only, no GUI backend
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
SRC = RESULTS / "gpu_profile_summary.json"
OUT = RESULTS / "gpu_host_overhead.png"

# Ops that move no data and compute nothing: dispatch, view bookkeeping and
# launch cost. Everything else is at least nominally real work.
OVERHEAD = {
    "cudaLaunchKernel", "aten::select", "aten::as_strided", "aten::empty",
    "cudaMemcpyAsync", "aten::copy_", "aten::reshape", "aten::transpose",
    "aten::unsqueeze", "aten::empty_strided",
}

GREEN, RED, BLUE, GREY = "#16a34a", "#dc2626", "#2563eb", "#cbd5e1"


def main() -> None:
    d = json.loads(SRC.read_text())
    m = d["meta"]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.55, wspace=0.22)

    # ---- occupancy over time ------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    g = np.array(d["bins_gpu"])
    lch = np.array(d["bins_launch"])
    t = np.arange(len(g)) * m["bin_us"] / 1000.0        # ms

    ax.fill_between(t, 0, g * 100, color=GREEN, alpha=0.85, label="GPU busy", zorder=3)
    ax.plot(t, lch * 100, color=BLUE, lw=0.9, alpha=0.9,
            label="host in cudaLaunchKernel", zorder=4)
    ax.axhline(m["busy_pct"], ls="--", lw=1.2, color=RED, zorder=5,
               label=f"mean GPU busy {m['busy_pct']:.1f}%")

    for i, b in enumerate(d["step_bounds"]):
        if b > 0:
            ax.axvline(b / 1000.0, color="#0f172a", lw=0.6, alpha=0.35,
                       zorder=2, label="decode step boundary" if i == 2 else None)

    ax.set(xlabel="time (ms)", ylabel="% of wall time", ylim=(0, 100),
           xlim=(0, t[-1]),
           title="The card is idle 85% of the time — it never clears ~22% busy in any window")
    ax.legend(fontsize=8, loc="upper right", ncol=4)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- time budget -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    busy_ms, idle_ms = m["gpu_busy_us"] / 1000, m["gpu_idle_us"] / 1000
    ax.barh([0], [busy_ms], color=GREEN, label=f"GPU busy  {busy_ms:.0f} ms")
    ax.barh([0], [idle_ms], left=[busy_ms], color=RED, alpha=0.75,
            label=f"GPU idle  {idle_ms:.0f} ms")
    ax.set(xlabel="ms across the profiled window", yticks=[],
           xlim=(0, busy_ms + idle_ms),
           title=f"{m['n_steps']} decode steps = "
                 f"{m['wall_us'] / 1000 / m['n_steps']:.0f} ms/step")
    ax.legend(fontsize=9, loc="center right")
    ax.spines[["top", "right", "left"]].set_visible(False)

    per_step = {
        "kernels launched": m["n_kernels"] / m["n_steps"],
        "CUDA API calls": m["n_launches"] / m["n_steps"],
        "ATen ops dispatched": m["n_cpu_ops"] / m["n_steps"],
    }
    txt = "\n".join(f"{v:>8,.0f}   {k}" for k, v in per_step.items())
    avg_gap = m["gpu_idle_us"] / max(m["n_kernels"], 1)
    launch = next(o for o in d["ops"] if o["name"] == "cudaLaunchKernel")
    txt += (f"\n{avg_gap:>8.1f}   µs idle per kernel, on average"
            f"\n{launch['self_ms'] * 1000 / launch['calls']:>8.2f}   "
            f"µs per cudaLaunchKernel")
    ax.text(0.01, -0.55, "per decode step\n" + txt, transform=ax.transAxes,
            fontsize=9, family="monospace", va="top")

    # ---- top CPU ops -------------------------------------------------------
    ax = fig.add_subplot(gs[1:, 1])
    ops = d["ops"][::-1]
    names = [o["name"] for o in ops]
    vals = [o["self_ms"] for o in ops]
    colors = [RED if n in OVERHEAD else GREEN for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.9)
    for i, o in enumerate(ops):
        ax.text(o["self_ms"] + 1.5, i, f"{o['calls']:,} calls",
                va="center", fontsize=7, color="#334155")
    ax.set_yticks(y, names, fontsize=8, family="monospace")
    ax.set(xlabel="self CPU time (ms)", xlim=(0, max(vals) * 1.35),
           title="Host time by op — red is pure overhead, green is real compute")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- kernels -----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ks = d["kernels"][::-1]
    kn = [k["name"].replace("void at::native::", "").replace("ampere_bf16_s16816", "")[:34]
          for k in ks]
    kv = [k["ms"] for k in ks]
    ax.barh(np.arange(len(kn)), kv, color=GREY, edgecolor="#64748b")
    ax.set_yticks(np.arange(len(kn)), kn, fontsize=7, family="monospace")
    ax.set(xlabel="GPU time (ms)",
           title=f"All GPU work in the window: {busy_ms:.0f} ms total")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Qwen3-4B decode on g5.2xlarge (A10G) — host-bound, not compute-bound",
        fontsize=13, y=0.975)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")

    print(f"\nGPU busy      {m['busy_pct']:.2f}% of {m['wall_us'] / 1000:.0f} ms")
    print(f"per step      {m['wall_us'] / 1000 / m['n_steps']:.1f} ms wall, "
          f"{m['gpu_busy_us'] / 1000 / m['n_steps']:.1f} ms of GPU work")
    print(f"kernels/step  {m['n_kernels'] / m['n_steps']:,.0f}")
    print(f"aten ops/step {m['n_cpu_ops'] / m['n_steps']:,.0f}")
    print(f"mean idle gap {avg_gap:.1f} us between kernels")


if __name__ == "__main__":
    main()

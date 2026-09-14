"""Before and after: what two changes did to a decode step on the A10G.

    uv run bench/day_5/plot_gpu_compare.py

Reads the two reduced trace summaries in bench/results/:

    gpu_profile_summary.json        the original code
    gpu_profile_summary_fast.json   torch.compile + SDPA/enable_gqa

A note on which numbers are honest, because the two sources disagree and it
matters. `busy_pct` inside those JSON files is computed over the *profiled*
window, and profiling with with_stack=True inflates wall time roughly 3x. The
steady-state ms/step below is measured by profile_decode.py before the profiler
starts, so it is clean. Everything per-step is therefore normalised as:

    wall      steady-state measurement, no profiler attached
    GPU work  profiler's total device time / 5 active steps
    busy      GPU work / wall

Counts are per active step (the schedule records 5), not per ProfilerStep marker
-- the trace emits two markers per step, which is why an earlier read of these
same files halved every count.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # file output only, no GUI backend
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "gpu_speedup.png"
ACTIVE_STEPS = 5

# Steady-state wall per decode step at 4 rows, from profile_decode.py.
WALL_MS = {"before": 88.23, "compile": 69.11, "after": 31.64}
# The middle run's device total, kept because it shows the cost I introduced.
COMPILE_GPU_MS_TOTAL = 340.637

RED, GREEN, BLUE, GREY = "#dc2626", "#16a34a", "#2563eb", "#cbd5e1"

# Inductor reports the whole compiled region as one op; it is not self time in
# the same sense as an aten op, so it would swamp the comparison.
SKIP_OPS = {"Torch-Compiled Region: 0/0"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    b = load("gpu_profile_summary.json")
    a = load("gpu_profile_summary_fast.json")

    gpu = {
        "before": b["meta"]["gpu_busy_us"] / 1000 / ACTIVE_STEPS,
        "compile": COMPILE_GPU_MS_TOTAL / ACTIVE_STEPS,
        "after": a["meta"]["gpu_busy_us"] / 1000 / ACTIVE_STEPS,
    }
    counts = {
        "kernels": (b["meta"]["n_kernels"] / ACTIVE_STEPS, a["meta"]["n_kernels"] / ACTIVE_STEPS),
        "CUDA API calls": (b["meta"]["n_launches"] / ACTIVE_STEPS, a["meta"]["n_launches"] / ACTIVE_STEPS),
        "ATen ops": (b["meta"]["n_cpu_ops"] / ACTIVE_STEPS, a["meta"]["n_cpu_ops"] / ACTIVE_STEPS),
    }

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.62, wspace=0.24)

    # ---- 1. where each step's time goes, all three versions ----------------
    ax = fig.add_subplot(gs[0, :])
    labels = ["before\n(eager)", "+ torch.compile\n(fixed-window clone)",
              "+ SDPA / enable_gqa\n(now)"]
    keys = ["before", "compile", "after"]
    y = np.arange(3)
    g = [gpu[k] for k in keys]
    h = [WALL_MS[k] - gpu[k] for k in keys]

    ax.barh(y, g, color=GREEN, label="GPU working", zorder=3)
    ax.barh(y, h, left=g, color=RED, alpha=0.75, label="GPU waiting on the host", zorder=3)
    ax.axvline(13.4, ls="--", lw=1.4, color="#0f172a", zorder=4,
               label="bandwidth floor 13.4 ms (8.04 GB / 600 GB/s)")
    for i, k in enumerate(keys):
        ax.text(WALL_MS[k] + 1.2, i, f"{WALL_MS[k]:.1f} ms/step   "
                f"{100 * gpu[k] / WALL_MS[k]:.0f}% busy", va="center", fontsize=9)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set(xlabel="ms per decode step at 4 rows", xlim=(0, 108),
           title="The GPU never got faster. It stopped waiting.")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ---- 2. GPU work is unchanged; host work collapsed ---------------------
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(2)
    gpu_pair = [gpu["before"], gpu["after"]]
    cpu_pair = [b["meta"]["wall_us"] * 0 + sum(o["self_ms"] for o in b["ops"]) / ACTIVE_STEPS,
                sum(o["self_ms"] for o in a["ops"] if o["name"] not in SKIP_OPS) / ACTIVE_STEPS]
    w = 0.36
    ax.bar(x - w / 2, gpu_pair, w, color=GREEN, label="GPU work")
    ax.bar(x + w / 2, cpu_pair, w, color=RED, alpha=0.8, label="host work")
    for i, (gv, cv) in enumerate(zip(gpu_pair, cpu_pair)):
        ax.text(i - w / 2, gv + 0.6, f"{gv:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, cv + 0.6, f"{cv:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["before", "after"])
    ax.set(ylabel="ms per step",
           title=f"GPU work {gpu_pair[1] / gpu_pair[0]:.2f}x, "
                 f"host work {cpu_pair[0] / cpu_pair[1]:.1f}x less")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- 3. how much less work the host issues ----------------------------
    ax = fig.add_subplot(gs[1, 1])
    names = list(counts)
    before = [counts[n][0] for n in names]
    after = [counts[n][1] for n in names]
    x = np.arange(len(names))
    ax.bar(x - w / 2, before, w, color=RED, alpha=0.8, label="before")
    ax.bar(x + w / 2, after, w, color=GREEN, label="after")
    for i, (bv, av) in enumerate(zip(before, after)):
        ax.text(i - w / 2, bv * 1.08, f"{bv:,.0f}", ha="center", fontsize=8)
        ax.text(i + w / 2, av * 1.08, f"{av:,.0f}", ha="center", fontsize=8)
        ax.text(i, max(bv, av) * 2.1, f"{bv / av:.1f}x fewer", ha="center",
                fontsize=8, color="#0f172a")
    ax.set_yscale("log")
    ax.set_xticks(x, names, fontsize=9)
    ax.set(ylabel="count per step (log)", ylim=(1, max(before) * 8),
           title="Per decode step, the host issues far less")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- 4. occupancy shape, before vs after -------------------------------
    ax = fig.add_subplot(gs[2, :])
    for d, color, lab in ((b, RED, "before"), (a, GREEN, "after")):
        bins = np.array(d["bins_gpu"])
        t = np.arange(len(bins)) * d["meta"]["bin_us"] / 1000
        t = t / t[-1] * 100                       # normalise: windows differ in length
        ax.plot(t, bins * 100, lw=1.0, color=color, alpha=0.9, label=lab)
        ax.fill_between(t, 0, bins * 100, color=color, alpha=0.18)
    ax.set(xlabel="profiled window (%)", ylabel="GPU busy in window (%)",
           xlim=(0, 100), ylim=(0, 100),
           title="Occupancy through the profiled window. Higher and denser is better; "
                 "note the profiler itself inflates wall time, so read the shape, not the level.")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Qwen3-4B decode on g5.2xlarge (A10G): 88.23 -> 31.64 ms/step, 2.79x",
                 fontsize=13, y=0.975)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}\n")

    print(f"{'':<16}{'before':>10}{'compile':>10}{'after':>10}")
    print(f"{'wall ms/step':<16}{WALL_MS['before']:>10.2f}{WALL_MS['compile']:>10.2f}{WALL_MS['after']:>10.2f}")
    print(f"{'GPU ms/step':<16}{gpu['before']:>10.2f}{gpu['compile']:>10.2f}{gpu['after']:>10.2f}")
    print(f"{'busy %':<16}{100*gpu['before']/WALL_MS['before']:>10.1f}"
          f"{100*gpu['compile']/WALL_MS['compile']:>10.1f}"
          f"{100*gpu['after']/WALL_MS['after']:>10.1f}")
    for n in names:
        print(f"{n:<16}{counts[n][0]:>10,.0f}{'':>10}{counts[n][1]:>10,.0f}")


if __name__ == "__main__":
    main()

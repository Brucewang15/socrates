"""Charts for the day-4 batching run. Reads bench/results/day4_batching.npz.

    uv run bench/day_4/batching.py
    uv run bench/day_4/plots.py

Four figures, each written to bench/results/:

    day4_timeline.png    per-job bars, one panel per policy
    day4_occupancy.png   rows busy over time, and rows held but idle
    day4_finish.png      when each job finished, sorted by output length
    day4_headline.png    throughput, TTFT and latency relative to static

Bars in day4_timeline match bench/day_3/batching.py:

    red dot   response time    caller sees its first token
    purple    turnaround       submission -> this job's last token
    grey      idle             finished, not yet handed back
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
SRC = RESULTS / "day4_batching.npz"

PURPLE, GREY, RED = "#7c3aed", "#e5e7eb", "#dc2626"
COLOR = {"static": "#2563eb", "continuous": "#16a34a"}


def load():
    if not SRC.is_file():
        raise SystemExit(f"{SRC} missing -- run: uv run bench/day_4/batching.py")
    z = np.load(SRC, allow_pickle=False)
    jobs = [str(j) for j in z["jobs"]]
    policies = [str(p) for p in z["policies"]]
    runs = {p: {k[len(p) + 1:]: z[k] for k in z.files if k.startswith(p + "_")}
            for p in policies}
    return z, jobs, policies, runs


def finish(fig, ax_or_axes, out) -> None:
    for ax in np.atleast_1d(ax_or_axes).ravel():
        ax.grid(alpha=0.25, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def timeline(jobs, policies, runs, max_batch, device) -> None:
    height = 0.30 * len(jobs) + 1.6
    fig, axes = plt.subplots(len(policies), 1, figsize=(11, height * len(policies)),
                             squeeze=False)
    xmax = max(runs[p]["delivered"].max() for p in policies) * 1.05
    y = np.arange(len(jobs))

    for ax, policy in zip(axes[:, 0], policies):
        m = runs[policy]
        ax.barh(y, m["delivered"], color=GREY, label="idle, waiting for the batch")
        ax.barh(y, m["own"], color=PURPLE, label="turnaround time")
        ax.plot(m["response"], y, "o", color=RED, ms=5, label="response time")
        ax.set_yticks(y, [j[:30] for j in jobs], fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("seconds since submission")
        ax.set_title(f"{policy} -- wall {m['wall']:.1f}s, "
                     f"{m['throughput']:.1f} tok/s, idle {m['idle_total']:.1f}s",
                     fontsize=9)
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{len(jobs)} jobs, {max_batch} rows, Qwen3-4B on {device}",
                 fontsize=10)
    finish(fig, axes, RESULTS / "day4_timeline.png")


def occupancy(policies, runs, max_batch) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for policy in policies:
        m = runs[policy]
        c = COLOR[policy]
        ax.step(m["occ_t"], m["occ_held"], where="post", lw=1.0, color=c, alpha=0.35)
        ax.fill_between(m["occ_t"], m["occ_live"], m["occ_held"], step="post",
                        color=c, alpha=0.12)
        ax.step(m["occ_t"], m["occ_live"], where="post", lw=1.6, color=c,
                label=f"{policy} (mean {float(m['mean_live']):.1f})")
    ax.axhline(max_batch, ls="--", lw=1, color="#334155", label=f"{max_batch} rows")
    ax.set(xlabel="time (s)", ylabel="rows", title="rows generating (solid) "
           "vs rows held (faint); the gap is wasted compute", ylim=(0, None))
    ax.legend(fontsize=8, loc="upper right")
    finish(fig, ax, RESULTS / "day4_occupancy.png")


def finish_times(jobs, policies, runs) -> None:
    order = np.argsort(runs[policies[0]]["tokens"])
    idx = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(jobs) + 1.8))
    h = 0.8 / len(policies)
    for i, policy in enumerate(policies):
        ax.barh(idx + (i - (len(policies) - 1) / 2) * h,
                runs[policy]["delivered"][order], height=h,
                color=COLOR[policy], label=policy)
    ax.set_yticks(idx, [f"{jobs[j][:26]} ({runs[policies[0]]['tokens'][j]})"
                        for j in order], fontsize=7)
    ax.set(xlabel="released (s)", title="when each job was handed back, "
           "sorted by tokens generated")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    finish(fig, ax, RESULTS / "day4_finish.png")


def headline(runs) -> None:
    if set(runs) != {"static", "continuous"}:
        print("skipped day4_headline.png -- needs both policies")
        return
    s, c = runs["static"], runs["continuous"]
    labels = ["throughput\n(tok/s)", "TTFT p99\n(s)", "latency p50\n(s)",
              "idle total\n(s)"]
    keys = ["throughput", "ttft_p99", "latency_p50", "idle_total"]
    fmts = ["{:.0f}", "{:.1f}", "{:.1f}", "{:.1f}"]
    sv = [float(s[k]) for k in keys]
    cv = [float(c[k]) for k in keys]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    # each metric has its own units, so normalise to static and print the real value
    ax.bar(x - 0.2, [1] * len(labels), width=0.4, color=COLOR["static"],
           label="static")
    ax.bar(x + 0.2, [b / a if a else 0 for a, b in zip(sv, cv)], width=0.4,
           color=COLOR["continuous"], label="continuous")
    for i, (a, b, fmt) in enumerate(zip(sv, cv, fmts)):
        ax.text(i - 0.2, 1.02, fmt.format(a), ha="center", fontsize=7)
        ax.text(i + 0.2, (b / a if a else 0) + 0.02, fmt.format(b), ha="center",
                fontsize=7)
    ax.axhline(1, lw=0.8, color="#334155")
    ax.set(xticks=x, ylabel="relative to static", title="headline numbers")
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    finish(fig, ax, RESULTS / "day4_headline.png")


def main() -> None:
    z, jobs, policies, runs = load()
    max_batch, device = int(z["max_batch"]), str(z["device"])
    print(f"{len(jobs)} jobs, {max_batch} rows, {device}, "
          f"policies: {', '.join(policies)}")

    timeline(jobs, policies, runs, max_batch, device)
    occupancy(policies, runs, max_batch)
    finish_times(jobs, policies, runs)
    headline(runs)


if __name__ == "__main__":
    main()

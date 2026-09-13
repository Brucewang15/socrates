"""Reduce a 114 MB PyTorch Chrome trace to a few KB of plottable summary.

Runs on the GPU host (the trace is too big to ship, and the instance role has
no s3:PutObject). Prints one line of JSON to stdout, small enough to come back
through SSM command output.

    python3 analyze_trace.py /tmp/prof/trace.json

Emitted:
  meta   window bounds, GPU busy vs wall, kernel and launch counts
  bins   GPU-busy fraction and host-launch fraction per time bin
  gaps   the largest GPU-idle intervals, and the CPU op covering each
  ops    top CPU ops by *self* time, computed by subtracting nested children
"""

import json
import sys
from collections import defaultdict

NBINS = 360
NGAPS = 12
NOPS = 14
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def merge(intervals):
    """Union of [start, end) intervals, sorted output."""
    if not intervals:
        return []
    intervals.sort()
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def self_times(events):
    """Self time per op name: duration minus time spent in nested children."""
    by_thread = defaultdict(list)
    for ev in events:
        by_thread[(ev.get("pid"), ev.get("tid"))].append(ev)

    self_us = defaultdict(float)
    total_us = defaultdict(float)
    counts = defaultdict(int)

    for evs in by_thread.values():
        # parents first: earlier start, and on ties the longer span encloses
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack = []  # [end, name]
        for ev in evs:
            ts, dur, name = ev["ts"], ev["dur"], ev["name"]
            while stack and stack[-1][0] <= ts:
                stack.pop()
            if stack:
                self_us[stack[-1][1]] -= min(dur, stack[-1][0] - ts)
            self_us[name] += dur
            total_us[name] += dur
            counts[name] += 1
            stack.append((ts + dur, name))

    return self_us, total_us, counts


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof/trace.json"
    with open(path) as f:
        trace = json.load(f)
    events = trace["traceEvents"]

    gpu, cpu_ops, runtime, steps = [], [], [], []
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev or "ts" not in ev:
            continue
        cat = ev.get("cat", "")
        if cat in GPU_CATS:
            gpu.append(ev)
        elif cat == "cpu_op":
            cpu_ops.append(ev)
        elif cat == "cuda_runtime":
            runtime.append(ev)
        name = ev.get("name", "")
        if name.startswith("ProfilerStep"):
            steps.append(ev)

    if not gpu:
        print(json.dumps({"error": "no GPU events in trace"}))
        return

    # Window: the span the profiler actually recorded on the device.
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["ts"] + e["dur"] for e in gpu)
    wall = t1 - t0

    busy = merge([[e["ts"], e["ts"] + e["dur"]] for e in gpu])
    busy_us = sum(e - s for s, e in busy)

    # Idle gaps between kernels, and which CPU op was on the host during each.
    cpu_sorted = sorted(cpu_ops, key=lambda e: e["ts"])
    gaps = []
    for (_, e_prev), (s_next, _) in zip(busy, busy[1:]):
        if s_next - e_prev <= 0:
            continue
        mid = (e_prev + s_next) / 2
        # deepest (shortest enclosing) cpu_op covering the middle of the gap
        cover = [c for c in cpu_sorted
                 if c["ts"] <= mid <= c["ts"] + c["dur"]]
        who = min(cover, key=lambda c: c["dur"])["name"] if cover else "(none)"
        gaps.append({"t": round(e_prev - t0, 1),
                     "d": round(s_next - e_prev, 1),
                     "op": who})
    gaps.sort(key=lambda g: -g["d"])
    idle_total = sum(g["d"] for g in gaps)

    # Binned occupancy for the timeline plot.
    bw = wall / NBINS
    gbins = [0.0] * NBINS
    for s, e in busy:
        i0, i1 = int((s - t0) / bw), min(int((e - t0) / bw), NBINS - 1)
        for i in range(i0, i1 + 1):
            lo, hi = t0 + i * bw, t0 + (i + 1) * bw
            gbins[i] += max(0.0, min(e, hi) - max(s, lo))
    lbins = [0.0] * NBINS
    for s, e in merge([[r["ts"], r["ts"] + r["dur"]] for r in runtime]):
        i0, i1 = int((s - t0) / bw), min(int((e - t0) / bw), NBINS - 1)
        for i in range(i0, i1 + 1):
            lo, hi = t0 + i * bw, t0 + (i + 1) * bw
            lbins[i] += max(0.0, min(e, hi) - max(s, lo))

    s_us, t_us, cnt = self_times(cpu_ops + runtime)
    top = sorted(s_us.items(), key=lambda kv: -kv[1])[:NOPS]

    kern_us = defaultdict(float)
    for e in gpu:
        kern_us[e["name"]] += e["dur"]
    top_kern = sorted(kern_us.items(), key=lambda kv: -kv[1])[:8]

    out = {
        "meta": {
            "wall_us": round(wall, 1),
            "gpu_busy_us": round(busy_us, 1),
            "gpu_idle_us": round(wall - busy_us, 1),
            "busy_pct": round(100 * busy_us / wall, 2),
            "n_kernels": len(gpu),
            "n_launches": len(runtime),
            "n_cpu_ops": len(cpu_ops),
            "n_steps": len(steps),
            "bin_us": round(bw, 3),
            "idle_in_gaps_us": round(idle_total, 1),
        },
        "step_bounds": [round(s["ts"] - t0, 1) for s in
                        sorted(steps, key=lambda e: e["ts"])][:12],
        "bins_gpu": [round(v / bw, 3) for v in gbins],
        "bins_launch": [round(v / bw, 3) for v in lbins],
        "gaps": gaps[:NGAPS],
        "ops": [{"name": n, "self_ms": round(v / 1000, 2),
                 "total_ms": round(t_us[n] / 1000, 2), "calls": cnt[n]}
                for n, v in top],
        "kernels": [{"name": n[:48], "ms": round(v / 1000, 2)} for n, v in top_kern],
    }
    print("JSON_BEGIN" + json.dumps(out, separators=(",", ":")) + "JSON_END")


if __name__ == "__main__":
    main()

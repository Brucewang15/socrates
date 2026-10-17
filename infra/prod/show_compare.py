"""Diff two /api/benchmark results -- ours against vLLM -- on the same axes.

    python3 show_compare.py /tmp/bench-engine.json /tmp/bench-vllm.json
    python3 show_compare.py /tmp/bench-paged.json /tmp/bench-vllm.json \\
        --baseline /tmp/bench-cont.json

With --baseline, a three-way table follows -- baseline, candidate, vLLM -- and
a PASS/FAIL on the one gate that decides a switch-over: the candidate's average
throughput must beat the baseline's, from the same session.

The headline is deliberately "ours as a percentage of vLLM", not an absolute
number: the absolute depends on the card, the prompt mix and the arrival rate,
while the ratio is the thing that says how much of a production engine's
throughput a hand-written one reaches.

Kept to %-formatting and no f-strings so it runs on the DLAMI's Python 3.10.
"""

import json
import sys

args = [x for x in sys.argv[1:]]
baseline_path = None
if "--baseline" in args:
    i = args.index("--baseline")
    baseline_path = args[i + 1]
    del args[i:i + 2]

if len(args) < 2:
    print(__doc__)
    raise SystemExit(2)

a = json.load(open(args[0]))              # ours
b = json.load(open(args[1]))              # vLLM
base = json.load(open(baseline_path)) if baseline_path else None


def same(key, path):
    """Guard against comparing two runs that did not run the same workload."""
    va, vb = a[path][key], b[path][key]
    return va, vb, ("" if va == vb else "  <-- DIFFERS")


print("workload")
for key in ("prompts", "rate", "seed", "max_tokens", "max_batch"):
    if key in a["context"] and key in b["context"]:
        va, vb, warn = same(key, "context")
        print("  %-14s ours %-10s vllm %-10s%s" % (key, va, vb, warn))
print("  %-14s ours %-10s vllm %-10s" % ("target", a["context"]["target"],
                                         b["context"]["target"]))
print("  %-14s ours %-10s vllm %-10s" % (
    "served", "Qwen3-4B", ",".join(b["context"].get("served") or ["?"])))

# Output token totals will not match exactly even at temperature 0: the two
# implementations differ in the last bits of the logits, so a few answers end a
# token or two earlier. A large gap means something real is different -- usually
# thinking mode left on, or a different max_tokens.
ta, tb = a["context"]["output_tokens"], b["context"]["output_tokens"]
skew = 100.0 * (ta - tb) / tb if tb else 0.0
print("  %-14s ours %-10d vllm %-10d (%+.1f%%)%s" % (
    "tokens made", ta, tb, skew,
    "  <-- more than 15% apart, check the workload matches" if abs(skew) > 15 else ""))

rows = [
    ("throughput avg tok/s", "throughput_tps", 1, "higher"),
    ("throughput p50 tok/s", "throughput_p50_tps", 1, "higher"),
    ("throughput peak tok/s", "throughput_peak_tps", 1, "higher"),
    ("per-stream tok/s", "per_stream_tps", 1, "higher"),
    ("TTFT p95 s", "ttft_p95_s", 2, "lower"),
    ("ITL p50 s", "itl_p50_s", 4, "lower"),
]

print()
print("%-22s %10s %10s %9s   %s" % ("metric", "ours", "vllm", "ours/vllm", "better"))
for label, key, dp, better in rows:
    va, vb = a["headline"][key], b["headline"][key]
    ratio = (100.0 * va / vb) if vb else float("nan")
    print("%-22s %10.*f %10.*f %8.1f%%   %s" % (label, dp, va, dp, vb, ratio, better))

print()
print("wall_s                 %10.1f %10.1f" % (a["context"]["wall_s"],
                                                b["context"]["wall_s"]))

# The number the writeup wants.
ours = a["headline"]["throughput_tps"]
theirs = b["headline"]["throughput_tps"]
print()
print("=> ours reaches %.1f%% of vLLM's aggregate throughput" % (100.0 * ours / theirs))
print("   (%.1f vs %.1f tok/s over %d prompts at %s/s)" % (
    ours, theirs, a["context"]["prompts"], a["context"]["rate"]))

# Per-second curves side by side, so the shape of the gap is visible rather than
# just its size: a lower plateau is a kernel/runtime story, a later ramp or a
# longer drain is a scheduler story.
sa, sb = a.get("throughput") or [], b.get("throughput") or []
if sa and sb:
    width = max(max(x["tokens_s"] for x in sa), max(x["tokens_s"] for x in sb)) or 1
    n = max(len(sa), len(sb))
    print()
    print("tokens/s per %gs bin        ours | vllm   (bar = whichever is longer)"
          % a["context"]["bin_s"])
    for j in range(n):
        xa = sa[j]["tokens_s"] if j < len(sa) else 0.0
        xb = sb[j]["tokens_s"] if j < len(sb) else 0.0
        ra = int(round(30 * xa / width))
        rb = int(round(30 * xb / width))
        print("  %4d %7.1f %7.1f  %-30s|%s" % (
            j, xa, xb, "#" * ra, "=" * rb))
    print("  # ours    = vllm")


# ---- three-way, when a baseline was given -----------------------------------
# The gate is relative to a run of the baseline from the same session on the same
# seeded load, never to a remembered number: day 7 measured two runs of the same
# code 0.1 tok/s apart, but the box, the driver and the prompt mix all drift
# between days.
if base is not None:
    na = base["context"].get("engine") or "baseline"
    nb = a["context"].get("engine") or "candidate"
    print()
    print("three-way, same session, same load")
    for key in ("prompts", "rate", "seed", "max_tokens"):
        if key in base["context"] and key in a["context"]:
            if base["context"][key] != a["context"][key]:
                print("  WARNING: %s differs: %s vs %s" % (key, base["context"][key],
                                                         a["context"][key]))
    print("  tokens made     %-8s %-8d %-8s %-8d vllm %d" % (
        na, base["context"]["output_tokens"], nb, a["context"]["output_tokens"],
        b["context"]["output_tokens"]))
    print()
    print("%-22s %10s %10s %10s %9s %9s %9s" % (
        "metric", na, nb, "vllm", nb + "/" + na, na + "/vllm", nb + "/vllm"))
    for label, key, dp, better in rows:
        x, y, z = base["headline"][key], a["headline"][key], b["headline"][key]
        r1 = 100.0 * y / x if x else float("nan")
        r2 = 100.0 * x / z if z else float("nan")
        r3 = 100.0 * y / z if z else float("nan")
        print("%-22s %10.*f %10.*f %10.*f %8.1f%% %8.1f%% %8.1f%%   %s" % (
            label, dp, x, dp, y, dp, z, r1, r2, r3, better))
    print("wall_s                 %10.1f %10.1f %10.1f" % (
        base["context"]["wall_s"], a["context"]["wall_s"], b["context"]["wall_s"]))

    x, y = base["headline"]["throughput_tps"], a["headline"]["throughput_tps"]
    delta = 100.0 * (y - x) / x if x else float("nan")
    print()
    print("=> gate: %s average throughput %.1f vs %s %.1f tok/s (%+.1f%%): %s" % (
        nb, y, na, x, delta, "PASS" if y > x else "FAIL"))
    if abs(delta) < 2.0:
        print("   within 2%: inside run-to-run noise -- re-run with another SEED before "
              "believing either direction")
    print("   share of vLLM: %s %.1f%% -> %s %.1f%%" % (
        na, 100.0 * x / theirs, nb, 100.0 * y / theirs))

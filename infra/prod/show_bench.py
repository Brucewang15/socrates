"""Print a /api/benchmark result as a table. Reads /tmp/bench.json on the host.

    python3 show_bench.py [path]

Kept as a file rather than an inline SSM one-liner because nested quotes in
f-strings need Python 3.12, and the DLAMI ships 3.10.
"""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench.json"
d = json.load(open(path))
rows = d["requests"]

hdr = ("bucket", "out", "queue", "prefill", "decode", "itl", "ttft", "total")
print("%-8s%5s%8s%8s%8s%9s%8s%8s" % hdr)
for r in rows:
    print("%-8s%5d%8.2f%8.2f%8.2f%9.4f%8.2f%8.2f" % (
        r["bucket"], r["output_tokens"], r["queue_s"], r["prefill_s"],
        r["decode_s"], r["itl_s"], r["ttft_s"], r["total_s"]))

itl = [r["itl_s"] for r in rows]
out = [r["output_tokens"] for r in rows]
print()
print("mean itl   %.4f s   (1/mean = %.2f tok/s)" % (
    sum(itl) / len(itl), 1 / (sum(itl) / len(itl))))
print("max  itl   %.4f s  on a request with %d output tokens" % (
    max(itl), out[itl.index(max(itl))]))
print("min  out   %d tokens" % min(out))
h, c = d["headline"], d["context"]
print("headline   throughput %.2f  peak %.2f  p50 %.2f  per_stream %.3f  occupancy %.3f" % (
    h["throughput_tps"], h["throughput_peak_tps"], h["throughput_p50_tps"],
    h["per_stream_tps"], h["occupancy"]))
print("context    %d tokens in %.1f s wall, over %s" % (
    c["output_tokens"], c["wall_s"], c.get("transport", "generate")))

# Throughput per bin, from measured token arrival times. The average above is
# one division; this is the shape it flattens, and the row count next to it is
# the explanation -- tokens/s tracks rows in the batch, not the clock.
series = d.get("throughput") or []
if series:
    width = max(1, max(b["tokens_s"] for b in series))
    total = sum(b["tokens"] for b in series)
    print()
    print("throughput per %.2fs bin   (sum %d tokens, %s output_tokens)" % (
        c.get("bin_s", 1.0), total,
        "matches" if total == c["output_tokens"] else "MISMATCH vs"))
    print("%7s %8s %6s %6s" % ("t", "tok/s", "rows", ""))
    for b in series:
        bar = "#" * int(round(40 * b["tokens_s"] / width))
        print("%7.2f %8.1f %6.1f %s" % (b["t"], b["tokens_s"], b["rows"], bar))

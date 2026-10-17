"""Paged decode kernel against the dense SDPA path, one attention layer at a time.

Runs on the GPU host, in the model image (infra/prod/kernel-on-instance.sh):

    python bench/day_8/kernel_bench.py --out /out/kernel_bench.json

Three things keep the numbers honest:

* Both paths run inside a captured CUDA graph, so launch overhead -- which the
  real decode step also does not pay -- is not counted against either.
* Each call reads a different layer's cache, cycling through LAYERS of them,
  so the L4's 48 MB L2 is cold the way it is in a real step, where 8 GB of
  weights pass through between two reads of the same layer's KV. Without this
  the realistic mix (~29 MB of live KV per layer) would sit in L2 and report
  bandwidth the card does not have.
* The dense path is the dense model's code, not a strawman: a bucketed window
  of K/V read as a strided view, the same keep-mask, SDPA with enable_gqa.

GB/s is computed from the bytes each path *must* read for its own design: live
KV for paged, the whole bucketed window for dense. "live GB/s" divides the live
bytes by each path's time, which is the comparison that matters -- the work done
is the same, only the bytes moved differ.
"""

import argparse
import json
import random

import torch
from torch.nn import functional as F

from model import kernels

HQ, HKV, D, BS = 32, 8, 128, 16
MAX_LEN, WIDTH = 2048, 128
LAYERS = 8
PEAK_GBS = 300.0
WINDOW_BUCKET = 512


def bucket(longest: int) -> int:
    need = longest + 1
    return min(MAX_LEN, max(WINDOW_BUCKET, -(-need // WINDOW_BUCKET) * WINDOW_BUCKET))


def setup(lens, seed=0):
    """LAYERS paged pools and LAYERS dense caches holding the same values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = len(lens)
    nb = B * WIDTH
    rng = random.Random(seed)
    phys = list(range(1, nb + 1))
    rng.shuffle(phys)
    table = torch.zeros(B, WIDTH, dtype=torch.int32)
    for b in range(B):
        table[b] = torch.tensor(phys[b * WIDTH:(b + 1) * WIDTH], dtype=torch.int32)
    table = table.cuda()

    pools, dense = [], []
    for _ in range(LAYERS):
        kp = torch.randn(nb + 1, HKV, BS, D, device="cuda", dtype=torch.bfloat16, generator=g)
        vp = torch.randn(nb + 1, HKV, BS, D, device="cuda", dtype=torch.bfloat16, generator=g)
        pools.append((kp, vp))
        # the dense cache is [B, MAX_LEN, HKV, D]: row b's slot t is at page t//BS
        bt = table.long()
        K = kp[bt].permute(0, 1, 3, 2, 4).reshape(B, WIDTH * BS, HKV, D).contiguous()
        V = vp[bt].permute(0, 1, 3, 2, 4).reshape(B, WIDTH * BS, HKV, D).contiguous()
        dense.append((K, V))
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, pools, dense, table, seq_lens


def dense_layer(q, K, V, positions, end):
    """qwen_kv_cont.Attention's read, for T=1."""
    k = K[:, :end].transpose(1, 2)
    v = V[:, :end].transpose(1, 2)
    keep = (torch.arange(end, device=q.device) <= positions[:, :, None])[:, None]
    return F.scaled_dot_product_attention(q[:, :, None], k, v, attn_mask=keep,
                                          enable_gqa=True)[:, :, 0]


def time_graph(fn, reps=20) -> float:
    """Microseconds per layer call, from a graph holding LAYERS x 4 calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()                                     # warm, and let allocators settle
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (reps * 4 * LAYERS)


def case(name, lens, partitions):
    q, pools, dense, table, seq_lens = setup(lens)
    scale = D ** -0.5
    positions = (seq_lens.long() - 1)[:, None]
    end = bucket(max(lens) - 1)
    live = sum(lens) * HKV * D * 2 * 2                # bytes, K and V
    window = len(lens) * end * HKV * D * 2 * 2

    # correctness on the spot, so a fast wrong kernel cannot win a row
    ref = dense_layer(q, *dense[0], positions, end).float()
    out = {"case": name, "B": len(lens), "mean_len": sum(lens) / len(lens),
           "max_len": max(lens), "window": end, "live_MB": live / 1e6,
           "window_MB": window / 1e6}

    def run_dense():
        for K, V in dense:
            dense_layer(q, K, V, positions, end)

    us = time_graph(run_dense)
    out["dense_us"] = us
    out["dense_GBs"] = window / us / 1e3
    out["dense_live_GBs"] = live / us / 1e3

    for P in partitions:
        got = kernels.paged_attention_decode(q, *pools[0], table, seq_lens, scale, P).float()
        err = (got - ref).abs().max().item()

        def run_paged(P=P):
            for kp, vp in pools:
                kernels.paged_attention_decode(q, kp, vp, table, seq_lens, scale, P)

        us = time_graph(run_paged)
        out[f"paged{P}_us"] = us
        out[f"paged{P}_GBs"] = live / us / 1e3
        out[f"paged{P}_err"] = err
    return out


def realistic(seed=0):
    """18 rows, mean ~390, longest in the 1024 bucket: day 7's measured regime."""
    rng = random.Random(seed)
    lens = [max(1, int(rng.expovariate(1 / 360))) for _ in range(17)]
    lens = [min(n, 1000) for n in lens] + [900]
    return lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out/kernel_bench.json")
    ap.add_argument("--partitions", default="0,128,256,512")
    args = ap.parse_args()
    partitions = [int(p) for p in args.partitions.split(",")]
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}", flush=True)

    cases = [("realistic", realistic())]
    for B in (1, 4, 8, 18):
        for n in (128, 390, 1024, 2048):
            cases.append((f"B{B}_L{n}", [n] * B))

    rows = []
    for name, lens in cases:
        r = case(name, lens, partitions)
        rows.append(r)
        best = min(partitions, key=lambda P: r[f"paged{P}_us"])
        print(f"{name:<12} B={r['B']:<2} mean={r['mean_len']:6.0f} win={r['window']:<4} "
              f"dense {r['dense_us']:7.1f}us ({r['dense_GBs']:5.0f} GB/s)  "
              + "  ".join(f"P{P} {r[f'paged{P}_us']:6.1f}us/{r[f'paged{P}_GBs']:4.0f}"
                          for P in partitions)
              + f"  best P{best} {r['dense_us'] / r[f'paged{best}_us']:.2f}x"
              f"  err {max(r[f'paged{P}_err'] for P in partitions):.1e}", flush=True)
        torch.cuda.empty_cache()

    real = rows[0]
    best = min(partitions, key=lambda P: real[f"paged{P}_us"])
    verdict = {
        "realistic_dense_us": real["dense_us"],
        "realistic_best_partition": best,
        "realistic_paged_us": real[f"paged{best}_us"],
        "realistic_speedup": real["dense_us"] / real[f"paged{best}_us"],
        "realistic_paged_pct_peak": 100 * real[f"paged{best}_GBs"] / PEAK_GBS,
        "realistic_dense_pct_peak": 100 * real["dense_GBs"] / PEAK_GBS,
        # x36 layers: what attention costs per decode step on each path
        "step_attention_ms_dense": real["dense_us"] * 36 / 1e3,
        "step_attention_ms_paged": real[f"paged{best}_us"] * 36 / 1e3,
        "gate_pass": real[f"paged{best}_us"] < real["dense_us"],
    }
    print("VERDICT " + json.dumps(verdict), flush=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "verdict": verdict, "partitions": partitions}, f, indent=2)


if __name__ == "__main__":
    main()

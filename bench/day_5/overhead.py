"""How much of a decode step is the GPU working, and how much is it waiting?

    uv run bench/day_5/overhead.py                 # on the g5
    uv run bench/day_5/overhead.py --batch 1 4 8   # the scaling test

Three measurements, cheapest first:

  1. sync count   -- how many times per step the CPU stalls on the GPU
  2. busy vs wall -- sum of kernel times over wall time. Well under 1.0 means
                    the GPU spent the step idle, waiting on us.
  3. scaling      -- if step time barely moves from batch 1 to batch 8, the
                    step is dominated by fixed overhead, not by work. That is
                    the signature of overhead-bound.
"""

import argparse
import time
import warnings

import torch

import model.inference_cont as cont


def build(batch: int):
    cont.MAX_BATCH = batch
    e = cont.Engine()
    for i in range(batch):
        e.submit("how to make pizza?")
    e.admit()                      # prefill, so decode_step has live rows
    return e


def sync_count(engine) -> int:
    """Count forced GPU->CPU synchronisations in one decode step."""
    if engine.model.embed_tokens.weight.device.type != "cuda":
        return -1                  # only CUDA reports these
    n = 0
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.decode_step()
        n = sum("synchron" in str(w.message).lower() for w in caught)
    torch.cuda.set_sync_debug_mode("default")
    return n


def timed_step(engine, n: int = 10) -> float:
    """Mean wall-clock seconds per decode step."""
    dev = engine.model.embed_tokens.weight.device
    for _ in range(3):
        engine.decode_step()       # warm up allocator and autotune
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        engine.decode_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()   # do not stop the clock mid-queue
    return (time.perf_counter() - t0) / n


def busy_fraction(engine) -> tuple[float, float]:
    """(kernel seconds, wall seconds) for one decode step, from the profiler."""
    from torch.profiler import ProfilerActivity, profile
    dev = engine.model.embed_tokens.weight.device
    acts = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        t0 = time.perf_counter()
        engine.decode_step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    attr = "self_device_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    busy = sum(getattr(e, attr, 0) for e in prof.key_averages()) / 1e6
    return busy, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    print(f"device {cont.DEVICE}\n")
    base = None
    for b in args.batch:
        e = build(b)
        step = timed_step(e)
        busy, wall = busy_fraction(e)
        syncs = sync_count(e)
        if base is None:
            base = step
        print(f"  batch {b:2d}  step {step * 1e3:7.2f} ms  "
              f"{step / base:4.2f}x batch-1  "
              f"busy {busy / wall:5.1%}  "
              f"syncs {syncs if syncs >= 0 else 'n/a'}")
        del e
        if cont.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n  step time flat across batch -> overhead-bound, not work-bound")
    print("  busy well under 100%          -> the GPU idled waiting on the CPU")


if __name__ == "__main__":
    main()

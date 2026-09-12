"""Continuous batching: a finished request is evicted and replaced the same step.

    uv run bench/day_3/continuous_batching.py
    uv run bench/day_3/continuous_batching.py --requests 32 --max-batch 8

Same model, same cache, same kernels, same requests as static_batching.py. The
only thing that changed is the admission rule in model/qwen_batch.Engine.admit:
rows are refilled every step instead of once per wave. So whatever gap the two
scripts show is scheduling and nothing else.

What to expect, and why:

  - occupancy holds near max_batch instead of sawtoothing down to 1, because a
    short request leaving frees a row that a waiting request takes immediately;
  - throughput rises roughly with mean batch size -- the weights get read once
    per step no matter how many rows ride along, so extra rows are nearly free
    while decode stays bandwidth-bound;
  - per-request latency for the *short* requests drops a lot, since they no
    longer queue behind a long request's whole wave.

Predict the throughput ratio from the two mean batch sizes before you look.

Results go to bench/results/continuous_batching.{npz,png}; compare.py reads the npz.
"""

import argparse

from workload import (
    add_args,
    build_requests,
    load_model,
    metrics,
    plot_run,
    print_metrics,
    print_workload,
    required_max_len,
    save,
)

from model.qwen_batch import Engine

STEM = "continuous_batching"


def main():
    ap = add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()

    tok, cfg, model, device = load_model(args)
    requests = build_requests(tok, args, device)
    max_len = required_max_len(requests)
    print_workload(requests, args.max_batch, max_len)

    engine = Engine(model, cfg, stop_ids=tok.all_special_ids,
                    max_batch=args.max_batch, max_len=max_len, policy="continuous")
    result = engine.run(requests, progress=True)

    m = metrics(result)
    print_metrics(m)
    print(f"\nfirst request's output: {tok.decode(requests[0].out_ids)[:200]!r}")

    save(result, STEM)
    plot_run(result, STEM, "#16a34a",
             f"Qwen3-4B, {device.type} {args.dtype}, max_batch={args.max_batch}")


if __name__ == "__main__":
    main()

"""Static batching baseline: a wave runs to completion before the next starts.

    uv run bench/day_3/static_batching.py
    uv run bench/day_3/static_batching.py --requests 32 --max-batch 8

This is the day-3 shape, and the thing continuous batching has to beat. All
requests in a wave are admitted together and nobody new gets in until the last
one finishes, so a 160-token request keeps a wave of 24-token requests open and
their rows sit idle. Watch the occupancy panel sawtooth down to 1.

Results go to bench/results/static_batching.{npz,png}; compare.py reads the npz.
"""

import argparse

from model.qwen.qwen_batch import Engine
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

STEM = "static_batching"


def main():
    ap = add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()

    tok, cfg, model, device = load_model(args)
    requests = build_requests(tok, args, device)
    max_len = required_max_len(requests)
    print_workload(requests, args.max_batch, max_len)

    engine = Engine(model, cfg, stop_ids=tok.all_special_ids,
                    max_batch=args.max_batch, max_len=max_len, policy="static")
    result = engine.run(requests, progress=True)

    m = metrics(result)
    print_metrics(m)
    print(f"\nfirst request's output: {tok.decode(requests[0].out_ids)[:200]!r}")

    save(result, STEM)
    plot_run(result, STEM, "#2563eb",
             f"Qwen3-4B, {device.type} {args.dtype}, max_batch={args.max_batch}")


if __name__ == "__main__":
    main()

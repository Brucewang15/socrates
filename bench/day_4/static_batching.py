"""Static batching baseline, on model/qwen_kv_seq.py.

    uv run bench/day_4/static_batching.py
    uv run bench/day_4/static_batching.py --requests 32 --max-batch 8

The wave model, and the thing continuous batching has to beat. A wave of
max_batch requests is padded to a common length, marched forward in lockstep on
one shared KV cache, and nobody new is admitted until every member is done. A
160-token request therefore keeps a wave of 24-token requests open, and their
rows keep being computed after they have nothing left to say.

Two consequences of the shared cache in model/qwen_kv_seq.KVCache, both worth
understanding rather than hiding:

  - one offset for the whole batch, so rows cannot be ragged. Prompts are
    left-padded to the wave's longest, exactly as backend/inference_static.py
    does it.
  - the mask is causal over shared positions with no padding mask, so a short
    prompt does attend to its own pad tokens and its real tokens sit at shifted
    RoPE positions. Its output can therefore differ from the same prompt run
    unpadded. That is a property of padded batching, and one of the arguments
    for the ragged cache in model/qwen_kv_cont.py.

Occupancy here is recorded as rows *still generating*, not the width of the
forward pass, so it is directly comparable to the continuous run. The gap
between the two is the waste: the tensor stays max_batch wide regardless.

Results go to bench/results/static_batching.{npz,png}; compare.py reads the npz.
"""

import argparse
import time

import torch
from workload import (
    add_args,
    build_requests,
    load_seq_model,
    metrics,
    plot_run,
    print_metrics,
    print_workload,
    save,
)

# Plain containers and a timing helper -- not the continuous scheduler.
from backend.inference_cont import RunResult
from model.qwen_kv_cont import sync
from model.qwen_kv_seq import KVCache

STEM = "static_batching"


def pad_wave(tok, wave, device):
    """Left-pad every prompt to the wave's longest, as inference_static.py does."""
    n = max(r.prompt_len for r in wave)
    pad = tok(" ").input_ids[0]
    return torch.cat(
        [torch.nn.functional.pad(r.input_ids, (n - r.prompt_len, 0), value=pad)
         for r in wave],
        dim=0,
    ).to(device), n


def run_static(model, cfg, tok, requests, *, max_batch, device, dtype) -> RunResult:
    """Waves of max_batch, lockstep until the slowest member of each is done."""
    stop_ids = set(tok.all_special_ids)
    step_t: list[float] = []
    step_batch: list[int] = []
    step_ms: list[float] = []
    step_kind: list[str] = []
    cache_bytes = 0
    padded_tokens = 0
    computed_rows = 0

    for req in requests:
        req.out_ids.clear()
        req.t_admit = req.t_first_token = req.t_done = None
        req.finish = ""

    sync(device)
    t_start = time.perf_counter()
    for req in requests:
        req.arrival = t_start

    for start in range(0, len(requests), max_batch):
        wave = requests[start:start + max_batch]
        admitted = time.perf_counter()
        for req in wave:
            req.t_admit = admitted

        ids, prompt_width = pad_wave(tok, wave, device)
        padded_tokens += sum(prompt_width - r.prompt_len for r in wave)
        room = prompt_width + max(r.max_new_tokens for r in wave) + 1
        cache = KVCache(len(wave), cfg["num_hidden_layers"], cfg["num_key_value_heads"],
                        cfg["head_dim"], max_len=room, dtype=dtype, device=device)
        itemsize = torch.empty((), dtype=dtype).element_size()
        cache_bytes = max(cache_bytes, 2 * cfg["num_hidden_layers"] * len(wave) * room
                          * cfg["num_key_value_heads"] * cfg["head_dim"] * itemsize)

        first = True
        with torch.no_grad():
            while not all(r.finish for r in wave):
                step = ids if len(cache) == 0 else ids[:, -1:]
                t0 = time.perf_counter()
                next_ids = model(step, cache)[:, -1].argmax(-1, keepdim=True)
                tokens = next_ids[:, 0].tolist()
                sync(device)
                t1 = time.perf_counter()

                live = sum(1 for r in wave if not r.finish)
                step_t.append(t0 - t_start)
                step_batch.append(live)          # rows doing useful work
                step_ms.append((t1 - t0) * 1000)
                step_kind.append("prefill" if first else "decode")
                computed_rows += len(wave)       # rows the forward actually paid for
                first = False

                for req, token in zip(wave, tokens):
                    if req.finish:
                        continue
                    if req.t_first_token is None:
                        req.t_first_token = t1
                    if token in stop_ids:
                        req.finish = "eos"
                        req.t_done = t1
                        continue
                    req.out_ids.append(token)
                    if req.n_generated >= req.max_new_tokens:
                        req.finish = "length"
                        req.t_done = t1

                # finished rows stay in the tensor; that is the point
                ids = torch.cat([ids, next_ids], dim=1)

        # the wave is only released when its slowest member lands
        released = time.perf_counter()
        for req in wave:
            if req.t_done is None:
                req.t_done = released

    sync(device)
    wall = time.perf_counter() - t_start
    result = RunResult(policy="static", requests=requests, wall=wall, step_t=step_t,
                       step_batch=step_batch, step_ms=step_ms, step_kind=step_kind,
                       cache_bytes=cache_bytes)
    result.padded_tokens = padded_tokens
    result.computed_rows = computed_rows
    return result


def main():
    ap = add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()

    tok, cfg, model, device = load_seq_model(args)
    requests = build_requests(tok, args, device)
    print_workload(requests, args.max_batch, max(r.prompt_len + r.max_new_tokens
                                                for r in requests))

    result = run_static(model, cfg, tok, requests, max_batch=args.max_batch,
                        device=device, dtype=getattr(torch, args.dtype))

    print_metrics(metrics(result))
    goodput = result.out_tokens / result.computed_rows
    print(f"  goodput             {goodput:.0%} of computed rows produced a token")
    print(f"  padding             {result.padded_tokens} pad tokens prefilled")
    print(f"\nfirst request's output: {tok.decode(requests[0].out_ids)[:200]!r}")

    save(result, STEM)
    plot_run(result, STEM, "#2563eb",
             f"Qwen3-4B, {device.type} {args.dtype}, max_batch={args.max_batch}")


if __name__ == "__main__":
    main()

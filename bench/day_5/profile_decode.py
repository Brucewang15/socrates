"""Profile one decode step of the continuous-batching engine on the GPU.

Stage 1 of 3. Runs *inside the model container on the GPU host*, not on your
laptop — it needs CUDA and the weights volume.

    # from your laptop, ship it up (the instance role can read this bucket)
    aws s3 cp bench/day_5/profile_decode.py s3://socrates-llm/ --profile management

    # on the host, via: aws ssm start-session --target <instance-id>
    aws s3 cp s3://socrates-llm/profile_decode.py /tmp/prof/profile_decode.py
    cd /opt/socrates
    docker compose stop model          # it holds the whole card
    docker compose run --rm -v /tmp/prof:/out -e PROF_OUT=/out \
      model python /out/profile_decode.py
    docker compose start model         # put the service back

Writes /out/trace.json (Chrome/Perfetto trace, ~114 MB), /out/table_cpu.txt,
/out/table_cuda.txt and /out/summary.json.

Then stage 2 is bench/day_5/analyze_trace.py on the host, and stage 3 is
bench/day_5/plot_gpu_profile.py on your laptop.
"""

import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule

import model.inference_cont as cont

OUT = os.getenv("PROF_OUT", "/out")
STEPS = 9  # wait 1 + warmup 3 + active 5

PROMPTS = [
    "how to make pizza?",
    "explain recursion",
    "how does a KV cache work?",
    "write a haiku about GPUs",
]


def main() -> None:
    print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.perf_counter()
    engine = cont.Engine()
    print(f"engine ready in {time.perf_counter() - t0:.1f}s device={cont.DEVICE}", flush=True)

    for p in PROMPTS:
        engine.submit(p)
    engine.admit()  # prefills each prompt into its own row
    print(f"rows live after prefill: {engine.n_active}", flush=True)

    # Warm up outside the profiler: first steps pay lazy init and autotune.
    for _ in range(5):
        engine.decode_step()
    torch.cuda.synchronize()

    # Wall-clock cost of a steady-state decode step, measured honestly.
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    for _ in range(20):
        engine.decode_step()
    torch.cuda.synchronize()
    wall_per_step_ms = (time.perf_counter() - w0) / 20 * 1000
    print(f"steady-state decode step: {wall_per_step_ms:.2f} ms wall", flush=True)

    sched = schedule(wait=1, warmup=3, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=lambda p: p.export_chrome_trace(f"{OUT}/trace.json"),
    ) as prof:
        for _ in range(STEPS):
            engine.decode_step()
            prof.step()

    def render(sort_key: str, **kw) -> str:
        try:
            return prof.key_averages(**kw).table(sort_by=sort_key, row_limit=30)
        except Exception as exc:  # key renamed across torch versions
            return f"[{sort_key} unavailable: {exc}]"

    cpu_table = render("self_cpu_time_total", group_by_stack_n=5)
    print(cpu_table, flush=True)

    cuda_table = render("self_device_time_total")
    if cuda_table.startswith("["):
        cuda_table = render("self_cuda_time_total")

    with open(f"{OUT}/table_cpu.txt", "w") as f:
        f.write(cpu_table)
    with open(f"{OUT}/table_cuda.txt", "w") as f:
        f.write(cuda_table)

    # Totals straight off the profiler, so the GPU-busy fraction is not guesswork.
    ev = prof.key_averages()
    cpu_us = sum(e.self_cpu_time_total for e in ev)
    dev_us = 0
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(ev[0], attr):
            dev_us = sum(getattr(e, attr) for e in ev)
            break

    summary = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows_live": engine.n_active,
        "wall_per_step_ms": wall_per_step_ms,
        "active_steps_profiled": 5,
        "self_cpu_us_total": cpu_us,
        "self_device_us_total": dev_us,
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

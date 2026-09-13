"""Time one forward pass at increasing prompt lengths. No generation.

    uv run bench/day_2/prefill.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from model.qwen.qwen import Qwen3, load_config, load_weights

LENGTHS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8126]
OUT = Path(__file__).resolve().parents[1] / "results" / "prefill.png"


def main():
    cfg = load_config()
    model = Qwen3(cfg)
    model.load_state_dict(load_weights(), strict=True)
    model = model.eval().to("mps", torch.bfloat16)

    times = []
    with torch.no_grad():
        for n in LENGTHS:
            ids = torch.randint(0, cfg["vocab_size"], (1, n), device="mps")
            for _ in range(2):        # first call compiles kernels; keep the second
                t0 = time.perf_counter()
                model(ids)[:, -1].argmax(-1).item()
                dt = time.perf_counter() - t0
            times.append(dt)
            print(f"n={n:5d}  {dt * 1000:8.0f} ms  {dt / n * 1000:6.2f} ms/token")

    plt.figure(figsize=(6, 4))
    plt.plot(LENGTHS, [t * 1000 for t in times], "o-", color="#2563eb")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("prompt length (tokens)")
    plt.ylabel("forward pass (ms)")
    plt.title("Qwen3-4B prefill, mps bf16")
    plt.grid(alpha=0.25, which="both")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

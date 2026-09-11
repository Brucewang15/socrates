"""Per-token latency of the from-scratch model with no KV cache: every step
re-runs the full forward pass over the whole sequence.

    uv run bench/day_2/no_kv_cache.py
"""

import time
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from model.qwen import MODEL_ID, Qwen3, load_config, load_weights
from transformers import AutoTokenizer

matplotlib.use("Agg")          # file output only, no GUI backend

PROMPT = "how to make pizza?"
OUT = Path(__file__).resolve().parents[1] / "results" / "no_kv_cache.png"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Qwen3(load_config())
    model.load_state_dict(load_weights(), strict=True)
    model = model.eval().to("mps", torch.bfloat16)

    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tok(text, return_tensors="pt").input_ids.to("mps")

    print(f"> {PROMPT}\n")
    lengths, latencies = [], []
    with torch.no_grad():
        while True:
            seq_len = ids.shape[1]
            t0 = time.perf_counter()
            next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
            token = next_id.item()          # blocks until the GPU is done
            latencies.append(time.perf_counter() - t0)
            lengths.append(seq_len)

            if token in tok.all_special_ids:
                break
            ids = torch.cat([ids, next_id], dim=1)
            print(tok.decode(next_id[0]), end="", flush=True)

    lengths, latencies = np.array(lengths), np.array(latencies) * 1000
    cumulative = np.cumsum(latencies) / 1000
    print(f"\n\n{len(latencies)} tokens | first {latencies[0]:.0f} ms | "
          f"last {latencies[-1]:.0f} ms | total {cumulative[-1]:.1f} s | "
          f"{len(latencies)/cumulative[-1]:.2f} tok/s")

    np.savez(OUT.with_suffix(".npz"), lengths=lengths, latencies=latencies)
    print(f"saved {OUT.with_suffix('.npz')}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(lengths, latencies, ".", ms=4, color="#2563eb")
    slope, intercept = np.polyfit(lengths, latencies, 1)
    ax1.plot(lengths, slope * lengths + intercept, "-", lw=1.2, color="#dc2626",
             label=f"fit: {slope:.3f}·n + {intercept:.0f} ms")
    ax1.set(xlabel="sequence length (tokens)", ylabel="latency per token (ms)",
            title="per-step cost grows linearly")
    ax1.legend(fontsize=8)

    ax2.plot(np.arange(1, len(cumulative) + 1), cumulative, lw=1.5, color="#2563eb")
    ax2.set(xlabel="tokens generated", ylabel="cumulative time (s)",
            title="total cost grows quadratically")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Qwen3-4B, no KV cache, mps bf16", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

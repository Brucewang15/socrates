# Day 2 — from-scratch forward pass, no KV cache

Wed Sep 10, 2026. Implemented `Qwen3-4B` in `model/qwen.py`, verified against
HuggingFace, then measured what generating without a KV cache actually costs.

---

## Memory, no cache

Assuming bf16 (2 bytes/value) and `n` = tokens in the current sequence.
All of this is **GPU** memory.

| | shape | size | lives |
|---|---|---|---|
| **weights** | — | **8.04 GB**, fixed | always resident |
| **hidden states** (residual stream) | `[1, n, 2560]` | 5 KB × n | between blocks |
| **q / k / v** | `[1, n, 4096]`, `[1, n, 1024]` ×2 | 12 KB × n | inside attention |
| **attention scores** | `[1, 32, n, n]` | **64 bytes × n²** | inside attention |
| **MLP intermediates** | 2–3 × `[1, n, 9728]` | 39–58 KB × n | inside the MLP |
| **logits** | `[1, n, 151936]` | **304 KB × n** | returned from forward |

# Day 2 — from-scratch forward pass, no KV cache

Wed Sep 10, 2026. Implemented `Qwen3-4B` in `model/qwen.py`, verified against
HuggingFace, then measured what generating without a KV cache actually costs.

---

## Memory, no cache

Assuming bf16 (2 bytes/value) and `n` = tokens in the current sequence.
All of this is **GPU** memory.


|                                     | shape                             | size               | lives                 |
| ----------------------------------- | --------------------------------- | ------------------ | --------------------- |
| **weights**                         | —                                 | **8.04 GB**, fixed | always resident       |
| **hidden states** (residual stream) | `[1, n, 2560]`                    | 5 KB × n           | between blocks        |
| **q / k / v**                       | `[1, n, 4096]`, `[1, n, 1024]` ×2 | 12 KB × n          | inside attention      |
| **attention scores**                | `[1, 32, n, n]`                   | **64 bytes × n²**  | inside attention      |
| **MLP intermediates**               | 2–3 × `[1, n, 9728]`              | 39–58 KB × n       | inside the MLP        |
| **logits**                          | `[1, n, 151936]`                  | **304 KB × n**     | returned from forward |


So, without a KV cache, the prefill step takes O(T^2) to calculate, which is dominated by the (n, n) attention pattern. However, this only starts being noticeable at high enough n

Every new token at length n+1 vs length n increases the compute time by o(n), as you need to compute 2n+1 entries in attention pattern, and a new MLP for the new token. For memory, it also scales by o(n)?

With a KV cache however, things are differnet.

## Memory, with KV cache — total at sequence length n

Every step feeds the model **one** token, so the activations stop scaling with n.


|                                     | shape                             | size at n         | at n = 1035 | lives           |
| ----------------------------------- | --------------------------------- | ----------------- | ----------- | --------------- |
| **weights**                         | —                                 | **8.04 GB** fixed | 8.04 GB     | always resident |
| **hidden states** (residual stream) | `[1, 1, 2560]`                    | 5 KB              | 5 KB        | freed each step |
| **q / k / v**                       | `[1, 1, 4096]`, `[1, 1, 1024]` ×2 | 12 KB             | 12 KB       | freed each step |
| **attention scores**                | `[1, 32, 1, n]`                   | 64 B × n          | 66 KB       | freed each step |
| **MLP intermediates**               | 2–3 × `[1, 1, 9728]`              | 39–58 KB          | 58 KB       | freed each step |
| **logits**                          | `[1, 1, 151936]`                  | 304 KB            | 304 KB      | freed each step |
| **KV cache**                        | 36 × 2 × `[1, n, 1024]`           | **144 KB × n**    | **149 MB**  | **persists**    |


Only two rows scale with n, and only one of them survives the step.

A token's query and key vectors will never change. For ex, for token 57, it can only look at tokens 0-57 (including itself) and can't read from future tokens due to casual masking. Hence after the forward pass at token 57, its query vector becomes useless, but future tokens can still use its key and value vectors.

At token 1000, prev key/value vectors cannot change so we don't have to include them at all! Hence only need to calculate the new query vector, new key vector and new value vector, and store the new key/value vector. hence why hidden state is only constant. This is called decoding.
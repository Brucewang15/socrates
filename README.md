# socrates

Build an LLM inference engine from first principles, in five days.

The rule for this project: **predict every number before you measure it.** Every
benchmark gets compared against arithmetic you did in advance. When the
prediction and the measurement disagree, that gap is the thing you go
investigate — and the reconciliation is the actual learning.

- **Implement** (days 1–2): `Qwen/Qwen3-4B` — dense, 4.022 B params, 8.04 GB @ bf16
- **Serve** (days 3–5): the same model, through a server you wrote yourself
- **Baseline:** vLLM, used as a yardstick to measure against — not as the thing
  you deploy
- **Hardware:** M4 Pro for correctness, `g6e.xlarge` (1× L40S, 48 GB, ~864 GB/s)
  for anything about speed
- **Deliverable:** `docs/writeup.md` — predicted vs. measured, with honest
  explanations of every gap

Why build the server instead of running vLLM: typing `vllm serve` teaches you
almost nothing — the learning is in the scheduler and the batching, and you
already have the hard part (a correct model with a KV cache). Thousands of
people have run vLLM; very few have implemented continuous batching.

Why the dense `Qwen3-4B` rather than `Qwen3.5-4B`: hand-writing gated delta-net
linear attention is a multi-day project on its own, and every transferable
lesson — RoPE, GQA, KV cache, pre-norm — lives in the canonical dense model.
`model/configs/` keeps the Qwen3.5 configs pinned for the day-1 arithmetic.

---

## Layout

```
analysis/    paper math + measurement scripts (start here)
model/       from-scratch PyTorch implementation (day 2)
  configs/   pinned config.json for each target model
engine/      the inference server — batching, scheduler, HTTP (days 3–5)
kernels/     CUDA kernels (optional stretch, only if time remains)
frontend/    chat UI (optional)
bench/       benchmark harness
  day_2/     latency with and without a KV cache
  results/   raw output + plots (gitignored)
infra/       EC2 launch + setup scripts
docs/        daily notes and the writeup — the real deliverable
```

`model/configs/` holds configs copied at a pinned revision, so the architecture
you reasoned about can't shift under you mid-week:

| file | repo | revision | params |
|---|---|---|---|
| `qwen3-4b.json` | `Qwen/Qwen3-4B` | `1cfa9a72…` | 4.022 B |
| `qwen3.5-4b.json` | `Qwen/Qwen3.5-4B` | `851bf6e8…` | 4.660 B |
| `qwen3.5-9b.json` | `Qwen/Qwen3.5-9B` | `c2022362…` | 9.653 B |
| `qwen3.5-27b.json` | `Qwen/Qwen3.5-27B` | `fc05daec…` | 27.781 B |
| `param_counts.json` | ground-truth cache | — | all of the above |

`analysis/arithmetic.py` reads these instead of hitting the Hub, so day 1 runs
with no network at all. Pass `--offline` to make that a hard guarantee. Anything
*not* pinned here still falls back to a Hub download, so you can point the script
at any model you like.

Two gotchas the loader handles for you, both worth understanding:

- Every Qwen3.5 repo nests the language model under a `text_config` key, because
  they're multimodal. Read the raw JSON and `hidden_size` won't be where you
  expect it.
- `tie_word_embeddings` is the exception: it lives at the *top* level, outside
  `text_config`. Unwrap naively and TODO(4) reads a key that isn't there.

---

## Reference: what `Qwen3.5-4B` actually is

Not a plain dense transformer. Know this before you write anything.

```
hidden_size          2560      num_attention_heads   16
intermediate_size    9216      num_key_value_heads    4     ← GQA, 4:1
num_hidden_layers      32      head_dim             256     ← NOT hidden/heads
vocab_size         248320      tie_word_embeddings true     ← lm_head is free
max_position_embeddings 262144
layer_types:  24 × linear_attention  +  8 × full_attention   ← hybrid
attn_output_gate: true    → q_proj is [8192, 2560], double the naive guess
```

| component | params | share |
|---|---|---|
| MLP (32 layers) | 2.265 B | 49% |
| linear attention (24 layers) | 1.011 B | 22% |
| embeddings | 0.636 B | 14% |
| vision tower | 0.334 B | 7% |
| full attention (8 layers) | 0.294 B | 6% |
| MTP head | 0.121 B | 3% |
| lm_head | 0 | tied to embeddings |
| **TOTAL** | **4.660 B** | |

Three kinds of GPU memory — never conflate them:

| | what | size | lifetime |
|---|---|---|---|
| **weights** | the checkpoint | fixed, **9.3 GB** @ bf16 | loaded once, shared by all requests |
| **KV cache** | K,V for seen tokens | **32 KB/token** (8 full-attn layers only) | per-request, grows with context |
| **linear state** | recurrent summary | **~50 MB/sequence, constant** | per-request, *does not grow* |
| activations | scratch inside a forward pass | small | microseconds |

That third row is unusual and important: you pay ~50 MB per concurrent request
before a single token of context. Classic "free VRAM ÷ bytes-per-token" capacity
math gives the wrong answer for this model. The crossover is around 1,500 tokens
— past that, the 24 linear layers are cheaper than 24 full-attention layers would
have been, and at the 262k max context they are cheaper by orders of magnitude.
That is the entire argument for hybrid attention, and you can see it in the
arithmetic.

**Fallback:** if vLLM's support for hybrid linear-attention models fights you,
switch to `Qwen/Qwen3-4B` (dense, previous generation, boringly well-supported).
Don't let a serving bug eat a whole day.

---

## Day 1 — Tue Sep 9: derive the limits on paper

**Goal:** know the model's memory and speed ceilings without touching a GPU, or
downloading a single weight.

1. Fill the five TODOs in `analysis/arithmetic.py`. Work against a *dense* model
   first, since that's what the script's formulas assume:
   ```
   uv run analysis/arithmetic.py Qwen/Qwen3-4B
   ```
   - TODO(1) embeddings: one `d`-wide vector per vocab entry.
   - TODO(2) attention per layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Q and O
     are sized by `num_attention_heads × head_dim`; K and V by
     `num_key_value_heads × head_dim`. A `Linear(in, out)` holds `in*out` weights.
     No biases in Qwen.
   - TODO(3) MLP per layer: SwiGLU is **three** matrices — `gate` and `up` are
     `d → d_ff`, `down` is `d_ff → d`. Guess whether attention or MLP dominates
     before you run it.
   - TODO(4) lm_head: zero if `tie_word_embeddings` is true, else `d × vocab`.
   - TODO(5) KV bytes/token: `2 (K and V) × N_layers × n_kv_heads × head_dim ×
     dtype_bytes`. The factor of 2 is the one people drop.

   Iterate until it prints `OK (<1% off)`. The 1% tolerance exists because the
   script ignores LayerNorm weights (~0.005% of the total).

2. Run it against all three Qwen3.5 sizes:
   ```
   uv run analysis/arithmetic.py Qwen/Qwen3.5-4B
   uv run analysis/arithmetic.py Qwen/Qwen3.5-9B
   uv run analysis/arithmetic.py Qwen/Qwen3.5-27B
   ```
   Every one prints **MISMATCH**, and that is correct behavior, not a bug in your
   math. Account for each difference by hand using the table above: the script
   models neither linear attention, nor the vision tower, nor the MTP head, and
   it assumes `q_proj` is half its real size.

   Compare 4B and 9B while you're here — 9B flips `tie_word_embeddings` to false,
   so `lm_head` costs a real 1.017 B, 10% of the model, from one boolean. That's
   TODO(4) earning its keep, and it costs you nothing to observe.

3. **Write your predictions down** in `docs/writeup.md`, for `Qwen3.5-4B` on one
   L40S (48 GB, 864 GB/s):
   - weight memory in GB
   - free VRAM after weights, assuming ~10% overhead
   - max concurrent requests at 4096 context — remember the 50 MB/sequence floor,
     it will dominate
   - batch-size-1 decode speed: `bandwidth ÷ weight_GB` tok/s

   These are the numbers everything else gets measured against. Committing them
   before you measure is the whole discipline. Predict the 27B numbers too while
   the reasoning is fresh — day 5 will want them.

4. Housekeeping so day 2 isn't a setup day:
   - Get an HF token, put it in `.env` as `HF_TOKEN=...` (already gitignored).
   - `git commit` the scaffold and your finished `arithmetic.py`.

**Done when:** the dense model prints OK, and your predictions are written down
and committed.

---

## Day 2 — Wed Sep 10: build the model yourself, then measure it

**Goal:** a forward pass you wrote, loading real weights, producing correct
logits — then the first real measurement.

**Morning, on your laptop** (M4 Pro, no cloud meter running):

1. `hf download Qwen/Qwen3-4B` (~8 GB). Use the *dense* previous-gen 4B for the
   from-scratch work — implementing gated delta-net linear attention on day 2 is
   how this project dies. You'll serve the hybrid 3.5-4B this afternoon; you do
   not need to have hand-written it.
2. Write `model/qwen.py`: RMSNorm, RoPE, GQA attention, SwiGLU MLP, the block,
   the stack. Read shapes from the config; your module attribute names must match
   the safetensors tensor names (`self_attn.k_proj.weight`) or `load_state_dict`
   will refuse.
3. **Verify against HuggingFace.** Same prompt, same seed, compare logits with
   `torch.allclose`. Do not proceed until they match. A wrong implementation that
   produces plausible-looking text will waste your entire week.
4. Greedy-decode 50 tokens with no cache. Time each step. Watch cost grow with
   sequence length — this is the O(n²) you're about to fix.
5. Add a KV cache. Verify identical output, then re-time. That curve going flat is
   the single most important plot in the project — save it.
6. Optional and cheap: `Qwen3.5-4B` is only 9.3 GB, so it fits your M4 too. Run
   it through HF `transformers` locally just to confirm the download and the
   hybrid architecture work before you're paying for a GPU.

**Afternoon, on AWS:**

7. Launch `g6e.xlarge`. **Give it a 200 GB gp3 EBS root volume** — the fast NVMe
   on this instance type is *instance store* and is wiped on every stop, and
   re-downloading each morning gets old. Point `HF_HOME` at EBS. Script the
   launch into `infra/` so you can rebuild it.
8. `hf download Qwen/Qwen3.5-4B` (~9.3 GB).
9. Run HF `generate()` at batch 1. Measure tok/s.
10. **Reconcile.** You predicted roughly `864 ÷ 9.3 ≈ 93` tok/s. You will measure
    less. Find out why: kernel launch overhead, Python overhead per step,
    sampling, attention that isn't purely weight-bound. Write the explanation
    down — this reconciliation is worth more than the number.

**Watch out:** ~$1.86/hr on-demand (us-east-1 — check current pricing). **Stop
the instance** whenever you step away. A forgotten weekend is ~$130.

**Done when:** your logits match HF, your KV cache works, and you can explain the
gap between predicted and measured tok/s.

---

## Day 3 — Thu Sep 11: batching

**Goal:** one forward pass serving several sequences at once.

Everything so far assumes `batch = 1`. Batching is the single most important
fact in LLM serving: 32 requests read the same 8.04 GB of weights once and get
32 tokens out of it. Same bytes moved, 32× the output. That's why decode being
bandwidth-bound is good news rather than bad.

1. **Generalise the KV cache to a batch.** `[B, n, 8, 128]` instead of
   `[1, n, 8, 128]`. This is where the `batch = 1` TODO in `KVCache` comes due.
2. **Handle ragged lengths.** Sequences in a batch have different lengths, so
   you need per-sequence position offsets for RoPE and a per-sequence mask. A
   padded batch where every sequence pretends to be the longest is the simple
   version; start there.
3. **Measure the sweep.** Batch 1 → 2 → 4 → 8 → 16 → 32 → 64. For each, record
   total throughput (tok/s), and p50 / p99 per-request latency.
4. **Plot throughput vs. batch size and explain the shape.** It should scale
   nearly linearly, then flatten. Predict where the knee falls *before* you
   plot it: batching stays nearly free while arithmetic intensity is below the
   GPU's FLOPs-per-byte ratio, and stops helping once you cross it.
5. **Push until you OOM.** Compare against the day-1 concurrency prediction.

**Done when:** a throughput-vs-batch curve, an explanation of its knee, and a
measured max batch size reconciled against the prediction.

---

## Day 4 — Fri Sep 12: continuous batching

**Goal:** requests join and leave the batch *every step* instead of waiting for
the slowest one to finish. This is the most interesting code you'll write all
week, and it's the main thing vLLM does that a naive server doesn't.

The problem with day 3's static batching: if one request wants 500 tokens and
seven want 20, the whole batch is held hostage for 500 steps and seven slots sit
idle. Under real traffic that wastes most of your GPU.

1. **A request queue.** Incoming prompts wait; finished ones are evicted.
2. **A scheduler.** Each step, decide which waiting requests get admitted, based
   on how much cache memory is free. Your day-1 arithmetic —
   `free VRAM ÷ 144 KB/token` — becomes actual admission-control code.
3. **Mid-batch join and leave.** A new request needs prefill while everyone else
   is decoding. The simple approach runs prefill separately, then merges; the
   better one interleaves them. Either is a legitimate design — document which
   you chose and why.
4. **Measure against day 3.** Same total work, uneven request lengths. The gap
   between static and continuous batching *is* the result.
5. **If time remains:** replace `torch.cat` with a preallocated buffer, then
   fixed-size blocks. At step 1000 you currently copy 149 MB to add 144 KB —
   that's the problem PagedAttention solves.

**Done when:** continuous batching measurably beats static batching under
uneven request lengths, and you can explain the mechanism.

---

## Day 5 — Sat Sep 13: serve it, compare it, write it up

**Morning — make it a real server.**

1. **FastAPI with token streaming.** An OpenAI-compatible
   `/v1/chat/completions` is worth the extra hour: every client already speaks
   it, and it makes the project demoable.
2. **Load-test it.** Concurrent clients, mixed prompt lengths, sustained
   traffic. Record throughput, p50/p99 latency, and time-to-first-token.

**Afternoon — vLLM as the yardstick.**

Install vLLM, serve the same model, run the identical load test, and report
**your engine as a percentage of vLLM's throughput.**

That's a better result than any absolute number, and it keeps the
industry-standard reference in the writeup without depending on it. Then name
the specific things that explain the gap — CUDA graphs, FlashAttention, fused
kernels, paged memory. Knowing *why* you're at 60% is worth more than being at
60%.

**Then `docs/writeup.md`. This is the deliverable.**

| quantity | predicted | measured | why they differ |
|---|---|---|---|
| weight memory | | | |
| batch-1 decode tok/s | | | |
| KV cache bytes/token | | | |
| max batch before OOM | | | |
| throughput @ batch 32 | | | |
| static vs. continuous batching | | | |
| % of vLLM throughput | | | |

A short section per gap explaining the mechanism. Honest gaps beat clean
numbers — "I predicted 93 tok/s, measured 61, and here is where the other 32
went" is a stronger signal than any benchmark.

**Last step:** terminate every instance. Check the EC2 console in every region
you touched.

---

## Optional stretch: one CUDA kernel

Only if the engine is done and working. Not a day; an afternoon.

A fused **RMSNorm** kernel is the cheapest way to keep the kernel signal —
simpler than SwiGLU, same argument. Write it in CUDA C++, bind it with
`torch.utils.cpp_extension`, verify numerics with `torch.allclose`, then
**predict the speedup before measuring**: count the HBM round-trips you
eliminated, multiply by tensor size, divide by 864 GB/s.

This matters more for kernel/compiler roles (Etched, Cerebras) than for
ML-infra roles generally. A working server beats a kernel attached to nothing,
so build the server first.

---

## Running tally of things to explain in the writeup

Keep notes as you go; these are the questions the project exists to answer.

- Why is decode memory-bandwidth-bound while prefill is compute-bound?
- Why does batching raise throughput without proportionally raising latency?
- What does GQA actually buy, in bytes?
- Why does the hybrid linear/full attention split change the capacity math, and
  where is the crossover?
- Why does `tie_word_embeddings` matter more at 4B than at 70B?
- Why does continuous batching beat static batching, and by how much?
- What is the scheduler actually deciding, and what constrains it?
- Why are Cerebras's tok/s numbers so high? (Hint: their weights live in on-wafer
  SRAM, so the bandwidth term in `bandwidth ÷ weight_GB` is replaced by something
  orders of magnitude larger. You will be able to answer this properly after
  day 2.)

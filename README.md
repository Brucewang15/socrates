# inference

Deploy a Qwen model on AWS from first principles, in five days.

The rule for this project: **predict every number before you measure it.** Every
benchmark you run gets compared against arithmetic you did in advance. When the
prediction and the measurement disagree, that gap is the thing you go investigate
— and the reconciliation is the actual learning.

- **Days 1–4 model:** `Qwen/Qwen3.5-4B` — 4.660 B params, 9.3 GB @ bf16
- **Day 5 model:** `Qwen/Qwen3.5-27B` — 27.781 B params, 55.6 GB @ bf16, does
  *not* fit one GPU, which is the point
- **Hardware:** `g6e.xlarge` (1× L40S, 48 GB, ~864 GB/s) for days 2–4;
  `g6e.12xlarge` or `g5.12xlarge` (4 GPUs) for day 5
- **Dates:** Day 1 = Tue Sep 9, 2026 → Day 5 = Sat Sep 13, 2026
- **Deliverable:** `docs/writeup.md` — a table of predicted vs. measured, with
  honest explanations of every gap

Why 4B and not 9B: they are architecturally identical in every dimension that
matters here — same 24/8 hybrid layer split, same 4 KV heads, same 256 head_dim,
same 32 KB/token KV cache, same ~50 MB/sequence linear state. 9B costs 2× to run
and teaches nothing extra. 4B also fits your M4 Pro, so correctness work costs
nothing. When you want a size that genuinely forces distributed systems, that's
27B on day 5, not 9B.

---

## Layout

```
analysis/    paper math + measurement scripts (start here)
model/       from-scratch PyTorch implementation (day 2)
  configs/   pinned config.json for each target model
kernels/     CUDA/C++ kernels (day 4)
backend/     serving layer — FastAPI in front of vLLM
frontend/    chat UI (day 5, optional)
bench/       load-test harness
  results/   raw output (gitignored)
infra/       EC2 launch + setup scripts
docs/        the writeup — this is the real deliverable
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

## Day 3 — Thu Sep 11: serve it properly

**Goal:** understand why batching is the entire economics of inference.

1. Install vLLM on the instance, serve `Qwen/Qwen3.5-4B`. Budget time for
   friction — hybrid linear-attention support is new. Fall back to dense
   `Qwen3-4B` if you lose more than half a day.
2. Compare vLLM at batch 1 to your day-2 number. It should be meaningfully
   faster; know which optimizations bought that (CUDA graphs, PagedAttention,
   fused kernels, continuous batching).
3. **The sweep.** Write `bench/` to drive concurrency 1 → 2 → 4 → 8 → 16 → 32 →
   64 → 128. At 9.3 GB of weights you have ~34 GB free, so you can push far past
   the knee — that headroom is exactly why 4B beats 9B here. For each level,
   record total throughput (tok/s), p50 and p99 per-request latency, and
   time-to-first-token. Save raw output to `bench/results/`.
4. Plot throughput vs. concurrency. Watch it scale nearly linearly, then flatten.
   **Explain the knee.** Below it you are memory-bandwidth-bound and batching is
   nearly free; above it you saturate something else. Which?
5. Push until you OOM. Compare against your day-1 prediction. The 50 MB/sequence
   linear-attention state is the term most likely to make you wrong — check
   whether it explains the gap.
6. Note the prefill/decode split: time-to-first-token is compute-bound prefill,
   inter-token latency is bandwidth-bound decode. They respond to batching
   completely differently, and conflating them is how people misread benchmarks.

**Done when:** you have a throughput-vs-concurrency curve, an explanation of its
knee, and a measured max concurrency reconciled against prediction.

---

## Day 4 — Fri Sep 12: go down a level (the differentiator)

**Goal:** one hand-written GPU kernel, benchmarked, and its speedup explained by
the same bandwidth arithmetic from day 1.

This is the day that separates the project from every other "I deployed an LLM"
repo. Anyone can run vLLM. Very few applicants have written a kernel and measured
it. If you are targeting Etched or Cerebras, **this day matters more than day 5**
— do not let the frontend steal it.

1. Pick one fusion target in `kernels/`. Easiest first:
   - **Fused RMSNorm** — simple, clearly bandwidth-bound, good first kernel.
   - **Fused SwiGLU** — `gate_proj`, `up_proj`, `silu`, multiply in one pass
     instead of three kernel launches and three round-trips to HBM. Bigger win,
     and the MLP is 49% of this model.
2. Write it in CUDA C++. Bind it with `torch.utils.cpp_extension`. Raw CUDA is
   higher signal than Triton for ASIC companies, because their kernel toolchains
   are C++-shaped.
3. **Verify numerics first** against the PyTorch version with `torch.allclose`,
   then drop it into your day-2 forward pass.
4. **Predict the speedup before benchmarking.** Count the HBM round-trips you
   eliminated, multiply by tensor size, divide by 864 GB/s. That gives you
   microseconds saved.
5. Benchmark with CUDA event timing and proper warmup. Report predicted vs.
   measured. Profile with `nsys` or `ncu` if the gap is large.

**Done when:** a kernel that is numerically correct, measurably faster, and whose
speedup you explained with arithmetic *before* you measured it.

---

## Day 5 — Sat Sep 13: distribute, then write it up

**Morning — scale to a model that doesn't fit.**

`Qwen/Qwen3.5-27B` is 27.781 B params → **55.6 GB @ bf16**, which does not fit a
single 48 GB L40S. Tensor parallelism stops being a demo and becomes a
requirement. Its shape:

```
hidden_size 5120   intermediate_size 17408   num_hidden_layers 64
num_attention_heads 24   num_key_value_heads 4   head_dim 256
layer_types: 48 × linear_attention + 16 × full_attention
KV cache: 64 KB/token        linear state: ~151 MB/sequence
```

1. Launch `g6e.12xlarge` (4× L40S, 192 GB) or `g5.12xlarge` (4× A10G, 96 GB).
2. Serve with `--tensor-parallel-size 2`, then 4. Measure throughput at each.
3. **Find where TP falls short of linear scaling.** TP=2 will not be 2× TP=1.
   That gap is all-reduce communication — every layer, both sublayers. Quantify
   it, and check whether NVLink vs. PCIe explains what you see.
4. If you'd rather do systems than kernels: skip TP and run two single-GPU
   replicas of 4B behind a load balancer. Compare round-robin against
   least-loaded routing under *uneven request lengths* — that's where you learn
   why head-of-line blocking is the defining problem of LLM serving.

**Afternoon — `docs/writeup.md`. This is the deliverable.**

Structure it as one table:

| quantity | predicted | measured | why they differ |
|---|---|---|---|
| weight memory | | | |
| batch-1 decode tok/s | | | |
| max concurrent @ 4096 ctx | | | |
| throughput @ batch 32 | | | |
| kernel speedup | | | |
| TP=2 speedup | | | |

Then a short section per gap explaining the mechanism. Honest gaps are worth more
than clean numbers — "I predicted 93 tok/s, measured 61, and here is where the
other 32 went" is a stronger signal than any benchmark you could report.

**If time remains:** `backend/` FastAPI wrapper with streaming, `frontend/` chat
UI. Be honest that these prove nothing about inference — build them only after
the writeup is done.

**Last step:** terminate every instance. Check the EC2 console in every region
you touched.

---

## Running tally of things to explain in the writeup

Keep notes as you go; these are the questions the project exists to answer.

- Why is decode memory-bandwidth-bound while prefill is compute-bound?
- Why does batching raise throughput without proportionally raising latency?
- What does GQA actually buy, in bytes?
- Why does the hybrid linear/full attention split change the capacity math, and
  where is the crossover?
- Why does `tie_word_embeddings` matter more at 4B than at 70B?
- What does tensor parallelism cost in communication, and why isn't TP=2 twice
  as fast?
- Why are Cerebras's tok/s numbers so high? (Hint: their weights live in on-wafer
  SRAM, so the bandwidth term in `bandwidth ÷ weight_GB` is replaced by something
  orders of magnitude larger. You will be able to answer this properly after
  day 2.)

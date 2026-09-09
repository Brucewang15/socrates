# inference

Deploy a Qwen model on AWS from first principles, in five days.

The rule for this project: **predict every number before you measure it.** Every
benchmark you run gets compared against arithmetic you did in advance. When the
prediction and the measurement disagree, that gap is the thing you go investigate
— and the reconciliation is the actual learning.

- **Target model:** `Qwen/Qwen3.5-9B`
- **Target hardware:** AWS `g6e.xlarge` (1× NVIDIA L40S, 48 GB HBM, ~864 GB/s)
- **Dates:** Day 1 = Tue Sep 9, 2026 → Day 5 = Sat Sep 13, 2026
- **Deliverable:** `docs/writeup.md` — a table of predicted vs. measured, with
  honest explanations of every gap.

---

## Layout

```
analysis/    paper math + measurement scripts (start here)
model/       from-scratch PyTorch implementation (day 2)
kernels/     CUDA/C++ kernels (day 4)
backend/     serving layer — FastAPI in front of vLLM
frontend/    chat UI (day 5, optional)
bench/       load-test harness
  results/   raw output (gitignored)
infra/       EC2 launch + setup scripts
docs/        the writeup — this is the real deliverable
```

---

## Reference: what `Qwen3.5-9B` actually is

Read from `config.json` and the safetensors headers. Not a plain dense
transformer — know this before you write anything.

```
hidden_size          4096      num_attention_heads   16
intermediate_size   12288      num_key_value_heads    4     ← GQA, 4:1
num_hidden_layers      32      head_dim             256     ← NOT hidden/heads
vocab_size         248320      tie_word_embeddings false    ← lm_head costs real params
max_position_embeddings 262144
layer_types:  24 × linear_attention  +  8 × full_attention   ← hybrid
attn_output_gate: true    → q_proj is [8192, 4096], double the naive guess
```

| component | params |
|---|---|
| MLP (32 layers) | 4.832 B |
| linear attention (24 layers) | 1.618 B |
| lm_head (untied) | 1.017 B |
| embeddings | 1.017 B |
| full attention (8 layers) | 0.470 B |
| vision tower | 0.456 B |
| MTP head | 0.243 B |
| **TOTAL** | **9.653 B** |

Three kinds of GPU memory — never conflate them:

| | what | size | lifetime |
|---|---|---|---|
| **weights** | the checkpoint | fixed, **19.3 GB** @ bf16 | loaded once, shared by all requests |
| **KV cache** | K,V for seen tokens | **32 KB/token** (8 full-attn layers only) | per-request, grows with context |
| **linear state** | recurrent summary | **~50 MB/sequence, constant** | per-request, *does not grow* |
| activations | scratch inside a forward pass | small | microseconds |

That third row is unusual and important: you pay ~50 MB per concurrent request
before a single token of context. Classic "free VRAM ÷ bytes-per-token" capacity
math gives the wrong answer for this model.

**Fallback:** if vLLM's support for hybrid linear-attention models fights you,
switch to `Qwen/Qwen3-8B` (dense, boringly well-supported). Don't let a serving
bug eat a whole day.

---

## Day 1 — Tue Sep 9: derive the limits on paper

**Goal:** know the model's memory and speed ceilings without touching a GPU.

1. Fill the five TODOs in `analysis/arithmetic.py`. Work against the *dense*
   model first, since that's what the script's formulas assume:
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

2. Re-run against the real target:
   ```
   uv run analysis/arithmetic.py Qwen/Qwen3.5-9B
   ```
   It will print **MISMATCH**, and that is correct behavior, not a bug in your
   math. Account for the difference by hand using the table above: the script
   models neither linear attention, nor the vision tower, nor the MTP head, and
   it assumes `q_proj` is half its real size. Write down what the script would
   need to change to handle a hybrid model. You don't have to implement it.

3. **Write your three predictions down** in `docs/writeup.md`, for a single L40S:
   - weight memory in GB
   - max concurrent requests at 4096 context (remember the 50 MB/sequence floor)
   - batch-size-1 decode speed: `bandwidth ÷ weight_GB` tok/s

   These are the numbers everything else gets measured against. Committing them
   before you measure is the whole discipline.

4. Housekeeping so day 2 isn't a setup day:
   - Get an HF token, put it in `.env` as `HF_TOKEN=...` (already gitignored).
   - `git commit` the scaffold and your finished `arithmetic.py`.

**Done when:** the dense model prints OK, and three predicted numbers are written
down and committed.

---

## Day 2 — Wed Sep 10: build the model yourself, then measure it

**Goal:** a forward pass you wrote, loading real weights, producing correct
logits — then the first real measurement.

**Morning, on your laptop** (M4 Pro, no cloud meter running):

1. `hf download Qwen/Qwen3-4B` (~8 GB). Use the *dense* 4B here — you are
   debugging numerics, and implementing gated delta-net linear attention on day 2
   is how this project dies.
2. Write `model/qwen.py`: RMSNorm, RoPE, GQA attention, SwiGLU MLP, the block,
   the stack. Read shapes from `config.json`; your module attribute names must
   match the safetensors tensor names (`self_attn.k_proj.weight`) or
   `load_state_dict` will refuse.
3. **Verify against HuggingFace.** Same prompt, same seed, compare logits with
   `torch.allclose`. Do not proceed until they match. A wrong implementation that
   produces plausible text will waste your entire week.
4. Greedy-decode 50 tokens with no cache. Time it. Watch cost grow with sequence
   length — this is O(n²).
5. Add a KV cache. Verify identical output, then re-time it. The shape of that
   curve going flat is the single most important plot in the project.

**Afternoon, on AWS:**

6. Launch `g6e.xlarge`. **Give it a 200 GB gp3 EBS root volume** — the fast NVMe
   on this instance type is *instance store* and is wiped on every stop, and
   re-downloading 19 GB each morning gets old fast. Point `HF_HOME` at EBS.
   Script the launch into `infra/` so you can rebuild it.
7. `hf download Qwen/Qwen3.5-9B` (~19.3 GB).
8. Run HF `generate()` at batch 1. Measure tok/s.
9. **Reconcile.** You predicted `864 ÷ 19.3 ≈ 45` tok/s. You will measure less.
   Find out why: kernel launch overhead, Python overhead per step, sampling,
   attention that isn't purely weight-bound. Write the explanation down.

**Watch out:** ~$1.86/hr on-demand (us-east-1 — check current pricing). **Stop
the instance** whenever you step away. A forgotten weekend is ~$130.

**Done when:** your logits match HF, your KV cache works, and you can explain the
gap between predicted and measured tok/s.

---

## Day 3 — Thu Sep 11: serve it properly

**Goal:** understand why batching is the entire economics of inference.

1. Install vLLM on the instance, serve `Qwen/Qwen3.5-9B`. Budget time for
   friction — hybrid linear-attention support is new. Fall back to `Qwen3-8B` if
   you lose more than half a day.
2. Compare vLLM at batch 1 to your day-2 number. It should be meaningfully
   faster; know which optimizations bought that (CUDA graphs, PagedAttention,
   fused kernels, continuous batching).
3. **The sweep.** Write `bench/` to drive concurrency 1 → 2 → 4 → 8 → 16 → 32 →
   64. For each, record total throughput (tok/s), p50 and p99 per-request
   latency, and time-to-first-token. Save raw output to `bench/results/`.
4. Plot throughput vs. concurrency. Watch it scale nearly linearly, then flatten.
   **Explain the knee.** Below it you are memory-bandwidth-bound and batching is
   nearly free; above it you saturate something else. Which?
5. Push concurrency until you OOM. Compare the number you get against your day-1
   prediction. The 50 MB/sequence linear-attention state is the term most likely
   to make you wrong — check whether it explains the gap.
6. Note the prefill/decode split: time-to-first-token is compute-bound prefill,
   inter-token latency is bandwidth-bound decode. They respond to batching
   completely differently.

**Done when:** you have a throughput-vs-concurrency curve, an explanation of its
knee, and a measured max concurrency reconciled against prediction.

---

## Day 4 — Fri Sep 12: go down a level (the differentiator)

**Goal:** one hand-written GPU kernel, benchmarked, and its speedup explained by
the same bandwidth arithmetic from day 1.

This is the day that separates the project from every other "I deployed an LLM"
repo. Anyone can run vLLM. Very few applicants have written a kernel and measured
it. If you are targeting Etched or Cerebras, this day matters more than day 5.

1. Pick one fusion target in `kernels/`. Good candidates, easiest first:
   - **Fused RMSNorm** — simple, clearly bandwidth-bound.
   - **Fused SwiGLU** — `gate_proj`, `up_proj`, `silu`, multiply in one pass
     instead of three kernel launches and three round-trips to HBM. Bigger win.
2. Write it in CUDA C++. Bind it with `torch.utils.cpp_extension`. Raw CUDA is
   higher signal than Triton for ASIC companies, because their kernel toolchains
   are C++-shaped.
3. **Verify numerics first** against the PyTorch version with `torch.allclose`,
   then drop it into your day-2 forward pass.
4. **Predict the speedup before benchmarking.** Count the HBM round-trips you
   eliminated, multiply by tensor size, divide by 864 GB/s. That gives microseconds
   saved.
5. Benchmark with proper CUDA event timing and warmup. Report predicted vs.
   measured. Profile with `nsys` or `ncu` if the gap is large.

**Done when:** a kernel that is numerically correct, measurably faster, and whose
speedup you explained with arithmetic *before* you measured it.

---

## Day 5 — Sat Sep 13: distribute, then write it up

**Morning — pick one** (both teach distributed systems; the first is deeper):

- **Tensor parallelism.** Launch `g5.12xlarge` (4× A10G), run vLLM with
  `--tensor-parallel-size 2`. Measure the speedup against TP=1 and find where it
  falls short of 2×. That gap is all-reduce communication cost over NVLink/PCIe.
  Quantify it.
- **Replica fleet.** Two single-GPU instances behind a load balancer. Compare
  round-robin against least-loaded routing under uneven request lengths. This is
  where you learn why head-of-line blocking matters for LLM serving.

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
than clean numbers — "I predicted 45 tok/s, measured 31, and here is where the
other 14 went" is a stronger signal than any benchmark you could report.

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
- Why does the hybrid linear/full attention split change the capacity math?
- Why does `tie_word_embeddings` matter more at 4B than at 70B?
- Why are Cerebras's tok/s numbers so high? (Hint: their weights live in on-wafer
  SRAM, so the bandwidth term in `bandwidth ÷ weight_GB` is replaced by something
  orders of magnitude larger. You will be able to answer this properly after
  day 2.)

# Day 1 — Transformer fundamentals + paper-math findings

Tue Sep 9, 2026. Concepts learned, then the numbers derived from `config.json`
alone. Corrections to my first-pass notes are marked with **⚠**.

---

# Part 1 — Concepts

## Tokens and embeddings

**Tokens** = the predefined vocabulary a model can input/output. Qwen3.5 has
248,320 of them.

**Embedding matrix** = each column is one token's vector. Part of model weights.

- Size: `(token dimension × # of tokens)` = `2560 × 248320`
- ⚠ That's the math convention (columns = tokens). On disk PyTorch stores it
  transposed as `[248320, 2560]` — `embed_tokens.weight`, one row per token.
  Same data, and `nn.Linear` weights are always `[out_features, in_features]`.

Transformers nowadays usually don't have biases, only weights. (True for the
Qwen language model, `attention_bias: false`. The vision tower *does* have
biases, and RMSNorm has a learned gain.)

## Softmax

Turns a vector of scores into a probability distribution.

- Values `0 ≤ x ≤ 1`, and they sum to 1
- `softmax(x_i) = e^(x_i) / Σ e^(x_j)`

## Unembedding matrix

Applied to the last token's vector to produce logits over the whole vocabulary.

- `W_U` size: `(# of tokens × token dimensions)` = `248320 × 2560`
- Outputs **logits** (one per vocabulary entry), the step before softmax
- ⚠ Applied after the final **block** - attention *and* MLP - and after a final
  RMSNorm. Not "after the final attention block."
- **Tied embeddings:** when `tie_word_embeddings: true`, `W_U` *is* the embedding
  matrix transposed - the same tensor, read the other direction. Costs zero
  extra parameters, and the checkpoint has no `lm_head.weight` at all.

## Temperature

Controls how flat or peaked the probability distribution is: `softmax(logits / T)`

- **T → 0**: deterministic, argmax, one probability goes to 1
- **T = 1**: default softmax, logits unchanged
- **T = 2**: divides logits by 2, so they end up closer together → flatter
  distribution → more random sampling

---

## Attention

**Job:** update each token's vector based on surrounding context.

**Idea:** for a word to update its meaning based on surrounding context, it first
needs to know which words before it are relevant.

### Query

- A query **VECTOR** is a small vector asking "what types of words am I looking
  for?" — 256 long (`head_dim`), vs the 2560-long embedding
- A query **MATRIX** is weights for **one head**. `W_Q [256, 2560]`, multiplied
  by one column of the prompt matrix (one token embedding) to give that token's
  query. In practice it's applied to all 400 columns in a single matmul.

### Key

- A key **VECTOR** is a small vector saying "I'm this type of word" — also 256
- ⚠ A key **MATRIX** is weights for **one head**, not one layer — same as query.
  (More precisely one *KV head*; see GQA below.)

### Value

- A value **MATRIX** is part of model weights, multiplied by every embedding
  vector to produce a value vector
- A token's raw embedding is a generic meaning. `W_V @ E_i` = what token `i`
  contributes **if another token attends to it**. The change is for the *other*
  token, not for itself.
- ⚠ The value vector is **256 long, not 2560** — same size as query and key.
- `update_vector_j = Σ_i ( attention_weight(j, i) × value_vector_i )`
- This weighted sum happens in 256-space. The update vector is then added to the
  original embedding to become a new embedding with its meaning updated by
  relevant surrounding context.

### ⚠ Output projection (`o_proj`) — missing from my first notes

The weighted sum above is 256 long. The residual stream is 2560. The step that
bridges them is `o_proj`, which 3B1B calls the **value↑ matrix**:

```
s_j    = Σ_i a_ji · v_i        [256]    weighted sum, in head space
Δ_j    = o_proj(s_j)           [2560]   lift to residual width
E_j   += Δ_j
```

`W_O [2560, 4096]` splits into 16 blocks of `[2560, 256]`, one per head. So:

```
concat(head_0..head_15) @ W_O  ==  W_O_0 @ head_0 + ... + W_O_15 @ head_15
```

Two jobs at once:

1. **Translation** — each head works in its own private 256-dim coordinate
   system; `W_O_h` is the learned dictionary from head `h`'s language into the
   residual stream's language.
2. **Combination** — the `+` signs. This is the **only** place heads mix.

**Why `o_proj` runs after the sum, not before.** It's linear, so
`o_proj(Σ a·v) = Σ a·o_proj(v)` — both orderings give the same answer. Doing it
after means one `o_proj` per query instead of one per cached token (~400× fewer
multiplies at 400 tokens), and it means the KV cache stores 256-wide vectors
instead of 2560-wide ones. Ordering is free; the savings are not.

### Attention pattern

Query · key dot product measures how similar two vectors are. If similar, the
query/key pair match and are semantically relevant.

- ⚠ Scores are divided by **√head_dim** (= 16 here) before softmax. Without it,
  dot products of 256-dim vectors get large, softmax saturates to near one-hot,
  and gradients vanish.
- **Causal masking:** future entries set to **−∞ before softmax**, which makes
  them 0 after. Masking after softmax would break normalization.
- Result: an `n × n` table where `n` = tokens in the sequence. Each query's
  weights over all keys form a probability distribution summing to 1.

⚠ **Convention warning — pick one and stick to it:**

| | 3B1B (what I learned) | PyTorch / papers / my day-2 code |
|---|---|---|
| score matrix | `Kᵀ Q` — rows keys, cols queries | `Q @ Kᵀ` — rows queries, cols keys |
| softmax over | each **column** | each **row** (`dim=-1`) |
| masked region | below the diagonal | above the diagonal |

Transposes of each other. **The invariant: softmax always runs along the KEY
axis**, so each query's weights sum to 1.

### Context window

Max number of tokens processed in one forward pass. `max_position_embeddings:
262144` — the name is a legacy artifact from when models had a learned
positional-embedding table. Qwen uses RoPE, which computes position instead of
looking it up, so no such table exists.

---

## Multi-headed attention

**Idea:** one head's Q/K/V matrices can only learn one type of contextual
relationship. Since a word can mean different things in different contexts, we
need several sets to capture that. Each pattern applies to ANY token.

```
E'_i = E_i + ΔE(1) + ΔE(2) + ... + ΔE(k)      k = heads in this block
```

### ⚠ GQA — missing from my first notes

`num_attention_heads` and `num_key_value_heads` are **different numbers**:

```
16 query heads    ← each token asks 16 different questions
 4 KV heads       ← only 4 sets of keys/values exist

query head  0 1 2 3   4 5 6 7   8 9 10 11   12 13 14 15
               ↓         ↓          ↓            ↓
kv head        0         1          2            3
```

Still 16 distinct attention patterns — what's shared is the material being
looked at, not the looking.

**Why shrink K/V and not Q:** queries are never cached (see below), so making
them cheaper saves nothing. Only K and V accumulate, so only they are worth
making smaller. Result: 4× smaller KV cache.

| | `n_kv_heads` | |
|---|---|---|
| MHA | = `n_heads` | every head has private K/V |
| **GQA** | **4** | groups share ← Qwen3.5 |
| MQA | 1 | all heads share one set |

---

## MLP

**Idea:** remember facts.

Processes each token **in parallel, with no context about other tokens**. All
cross-token communication already happened in attention. This is why GPUs
matter — 400 independent passes become one big matmul.

**3 layers, 2 matrices** (classic version): `2560 → 9216 → 2560`. One hidden
layer, ~3.6× wider. Depth comes from stacking 32 blocks, not from deep MLPs.

- **`up_proj`** `(kn × n)`, n = embedding space: each row maps the first layer to
  one entry of the second. Asks a question — if the dot product is positive, the
  answer is true. # of rows = # of questions asked.
- **`down_proj`** `(n × kn)`: each column is the answer written back.
- Output is added to the original vector (residual).
- **ReLU** = `max(0, x)`. Can't update `x` based on other information — one input
  to one output.

**The MLP is position-blind.** It receives a 2560-vector and has no idea whether
it came from position 7 or 300. All positional info entered via RoPE, in
attention.

### SwiGLU

Swish (a.k.a. SiLU, `x · sigmoid(x)`) is a smooth version of ReLU.

```
input → gate_proj and up_proj (parallel) → ⊙ → down_proj → output
```

- `gate_proj` passes through Swish; `up_proj` does not. Both output a wider
  vector (9216).
- ⚠ **`result = SiLU(gate_vector) ⊙ up_vector` — element-wise MULTIPLY, not
  add.** That's the entire meaning of "gated": `gate` decides *whether and how
  strongly* each of the 9216 neurons fires, `up` decides *what* it contributes.
  Addition would collapse two parallel linear paths into one linear map and
  destroy the point.
- Then `final = down_proj @ result`, added to the original input embedding.

**Why 3 matrices when there are only 3 layers?** In a plain feedforward net,
`#matrices = #layers − 1`. A GLU splits the input→hidden edge into two parallel
branches, so you get one extra. Still one hidden layer.

Historically classic MLPs used `d_ff = 4d` with 2 matrices (`8d²`); SwiGLU models
shrank to `d_ff ≈ 2.67d` with 3 matrices (also `8d²`) — same budget, better
quality. **Qwen didn't shrink much**: `d_ff = 9216` against `d = 2560` is 3.6×,
so ~`10.8d²`. That's why the MLP is 49% of this model.

---

## KV cache

**Why it exists:** naively, generating token 401 re-runs the forward pass over
all 401 tokens, recomputing K and V for tokens 1–400 that are bit-for-bit
identical to last step's. Causal masking guarantees they can never change —
token 57's key depends only on tokens ≤ 57, at every layer. So store them.

- ⚠ **Query vectors are not cached because nothing ever reads them again.** A
  query is consumed at its own step — it asked its question, got its answer, and
  produced the next token. Causal masking is a separate fact: it's what makes K
  and V stay *valid*, not what makes Q useless.
- ⚠ Key/value vectors can be cached because earlier tokens' representations are
  **unchanged** by later tokens — not "reset."
- **Every layer has its own cache.** 32 layers → 32 separate K/V caches, all
  live, all appended to every step. That's where the `× n_layers` comes from.
- Cached vectors are the **pre-`o_proj` 256-wide** ones.
- A token stores only **its own** K and V — nothing about who it attends to.
  That's why the per-token cost is constant regardless of position.

```
bytes per token = n_layers × 2 × n_kv_heads × head_dim × dtype_bytes
                              ↑        ↑          ↑
                          K and V   NOT n_heads   width of each
```

### Complexity

| | per token added | total over n tokens |
|---|---|---|
| **memory** | **constant** (32 KB) | **O(n)** linear |
| **compute** | **O(i)** — new query @ all i prev keys | **O(n²)** quadratic |

Constant marginal cost sums to linear. Growing marginal cost sums to quadratic.

- Prefill computes the full `[T,T]` pattern once → O(T²) compute, but the cache
  it fills is O(T). The `[T,T]` table is a transient **activation**, and
  FlashAttention tiles it so it's never written to HBM.
- Each decode step computes only **one new row** `[1, T]`, not the whole table.
  Without a cache you'd rebuild the table each step: O(n³) total.
- The cache itself doesn't compute anything — it's storage. Memory grows
  linearly, attention compute grows quadratically. Two separate budgets.

### Weights vs activations

| | weights | attention pattern / activations |
|---|---|---|
| from | training | this specific input |
| changes per prompt | never | every time |
| lives in | the 9.3 GB checkpoint | scratch memory, then discarded |
| size | fixed | grows with sequence length² |

---

# Part 2 — Day 1 findings

## `analysis/arithmetic.py` validated on a dense model

`Qwen/Qwen3-4B` (dense, no vision, no MTP, 36 layers, 8 KV heads, head_dim 128):

```
embeddings    0.389 B    9.7%
attention     0.944 B   23.5%
mlp           2.690 B   66.9%
lm_head       0.000 B    0.0%   ← tied
TOTAL         4.022 B
actual        4.022 B   <- OK (0.0% off)
```

Every parameter accounted for from 2 KB of JSON. The remaining 0.005% is
LayerNorm gains, which the script ignores.

### Formulas

```
embeddings     = d × vocab
attn_per_layer = (n_heads + n_kv_heads) × 2 × hd × d
mlp_per_layer  = 3 × d × d_ff
lm_head        = 0 if tie_word_embeddings else d × vocab
kv_per_token   = n_layers × 2 × n_kv_heads × hd × dtype_bytes
```

## `Qwen3.5-4B` — MISMATCH, and why

The script prints **19.8% off** (3.739 B vs 4.660 B). This is the script being
honest, not broken: it assumes every model is a dense transformer.

| script says | truth | |
|---|---|---|
| embeddings 0.636 B | 0.636 B | ✓ |
| **attention 0.839 B** | see below | ✗ |
| mlp 2.265 B | 2.265 B | ✓ |
| lm_head 0.000 B | 0 (tied) | ✓ |

Real breakdown:

| component | params | share |
|---|---|---|
| MLP (32 layers) | 2.265 B | 49% |
| linear attention (24 layers) | 1.011 B | 22% |
| embeddings | 0.636 B | 14% |
| vision tower | 0.334 B | 7% |
| full attention (8 layers) | 0.294 B | 6% |
| MTP head | 0.121 B | 3% |
| **TOTAL** | **4.660 B** | |

Reconciliation:

```
 3.739   script total
-0.839   its wrong attention line
+0.294   real full attention (8 layers, gated q_proj)
+1.011   linear attention (24 layers)
+0.334   vision tower
+0.121   MTP head
──────
 4.660   ✓
```

### Three reasons the attention line is wrong

**1. Only 8 of 32 layers have `self_attn`.** `layer_types` alternates
`linear_attention × 3, full_attention × 1` — 24 linear + 8 full. The other 24
use gated delta-net linear attention with a fixed-size recurrent state.

**2. `attn_output_gate: true` doubles `q_proj`.** Expected `16 × 256 = 4096`
rows; actual tensor is `[8192, 2560]`. The extra 4096 rows produce a gate on the
attention output.

Per full-attention layer, from real tensor shapes:

```
q_proj [8192, 2560] = 20,971,520
k_proj [1024, 2560] =  2,621,440
v_proj [1024, 2560] =  2,621,440
o_proj [2560, 4096] = 10,485,760
                      ──────────
                      36,700,160 × 8 layers = 0.294 B
```

**3. Vision tower and MTP head aren't modeled at all.** Every Qwen3.5 variant is
natively multimodal (`Qwen3_5ForConditionalGeneration`), including `-Base`.
There is no text-only Qwen3.5.

### `head_dim` is decoupled from `hidden_size`

`16 × 256 = 4096 ≠ 2560`. The attention block deliberately works in a *wider*
space than the residual stream it reads from — which is why `o_proj` is
`[2560, 4096]`. Dividing `hidden_size / num_attention_heads` gives 160 and is
37% wrong. Always read `head_dim` from the config.

## Predictions for day 2 — `Qwen3.5-4B` on `g6e.xlarge` (L40S, 48 GB, 864 GB/s)

Corrected by hand, since the script's numbers assume a dense model:

| quantity | predicted | how |
|---|---|---|
| **weight memory** | **9.32 GB** | 4.660 B × 2 bytes (bf16) |
| **free VRAM** | **33.9 GB** | 48 × 0.9 − 9.32 |
| **KV cache / token** | **32 KB** | 8 full-attn layers × 2 × 4 kv heads × 256 × 2 B |
| **linear state / sequence** | **~50 MB** | 24 layers × 32 value heads × 128 × 128 × 4 B, constant |
| **concurrent @ 4096 ctx** | **~187** | 33.9 GB ÷ (4096 × 32 KB + 50 MB) |
| **batch-1 decode** | **92.7 tok/s** | 864 ÷ 9.32 |

⚠ The script's own output for this model (`128 KB/token`, `26 concurrent`,
`80.2 tok/s` on A10G) is wrong on all three counts — it counts 32 caching layers
instead of 8, ignores the linear-attention state, and undercounts weights.

### Why the hybrid split matters

Per-sequence memory at 4096 context is **50 MB of constant linear state + 131 MB
of KV cache**. The state floor dominates at short context; the cache dominates at
long. Crossover ≈ 1,500 tokens.

Second crossover, on decode speed. Each step moves weights (fixed) plus KV cache
reads (growing):

```
Qwen3-4B  (dense):  8.04 GB ÷ 144 KB/token ≈  56,000 tokens
Qwen3.5-4B (hybrid): 9.32 GB ÷  32 KB/token ≈ 284,000 tokens
```

Past that length, attention moves as many bytes as the weights and token rate
halves. For the hybrid model that point is **beyond its 262k max context** —
attention never dominates decode at any supported length. That is what the
architecture bought, in one number.

## Things to check on day 2/3

- Measured tok/s vs the 92.7 ceiling — expect less; find where it goes
- Measured max concurrency vs ~187 — the 50 MB/seq floor is the likeliest
  source of error
- Whether vLLM lets me skip loading the vision tower (0.67 GB back for cache)
- Whether MTP speculative decoding is supported — it's the one legitimate way to
  **beat** the 92.7 tok/s ceiling, by verifying several guessed tokens in one
  weight-streaming pass
- Decode tok/s at 1k / 8k / 32k context, to see where the curve bends

## Not learned yet — gaps for day 2

- **RoPE** — how position actually enters the model
- **RMSNorm** — the normalization before each sublayer
- **Gated delta-net / linear attention** — what those 24 layers actually do
- **FlashAttention** — how the `[T,T]` pattern is tiled and never materialized
- **PagedAttention** — how vLLM manages cache memory
- **Speculative decoding** — how MTP would be used at inference

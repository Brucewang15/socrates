Day 8 is paged attention: the KV cache stops being one rectangle per row and
becomes a pool of 16-token blocks that rows borrow, and decode attention becomes
a hand-written CUDA kernel that walks each row's own blocks and nothing else.

Day 7 ended with the reason. Every row in a decode step read the same KV window,
sized to the longest live row rounded up to a 512-slot bucket, so a 60-token
answer paid for the 1,000-token one beside it. The step is bandwidth-bound, so
those padding bytes are time.

Same box as always: g6.2xlarge, one L4 (24 GB, ~300 GB/s), Qwen3-4B in bf16,
18 rows of 2048 context, 64 prompts arriving Poisson at 2/s, seed 0.

## The prediction, written down before anything was measured

The step reads weights once and the KV window once per layer:

    weights                                              8.04 GB
    KV, bucketed: 18 rows x 1024 slots x 144 KB          2.72 GB    (day 7's mean bucket)
                                                        --------
                                                        10.76 GB    -> 64.2 ms measured

A row's context averaged over its life is its prompt plus half its answer, which
for this load is ~300 tokens, so the bytes that are actually live are

    KV, live: 18 rows x ~300 slots x 144 KB              0.80 GB
                                                        --------
                                                         8.84 GB

At the 55% of peak bandwidth day 7 measured, that is a 53.6 ms step instead of
64.2 ms: **1.20x on ITL, and close to +20% throughput in the saturated part of
the run.** The run average moves less than that, because the ramp and the drain
are not saturated and gain proportionally less.

Two ways this loses, both worth stating before the numbers can argue:

1. **Kernel efficiency.** The weights still go through cuBLAS at whatever rate
   they go now; only attention changes hands. Today attention moves 2.72 GB in
   about 16.5 ms. The paged kernel moves 0.80 GB, so it breaks even at
   0.80 / 16.5 ms = 48 GB/s -- **16% of peak**. Anything better is a win. That is
   a lot of margin, but a first kernel with uncoalesced loads can be slower than
   that.
2. **Occupancy when the batch drains.** One thread block per (row, kv head) is
   18 x 8 = 144 blocks at full batch and 8 at one row, on 58 SMs. SDPA splits
   the KV read across blocks to keep the card busy; a kernel that doesn't is
   idle exactly in the drain. That is what split-K (flash-decoding) is for, and
   it is in the kernel from the start rather than retrofitted.

And one thing that comes for free: `WINDOW_BUCKET` goes away. The kernel reads
each row's length on the device, so the launch shape no longer depends on the
data, and the 18 x 4 = 72 captured graphs become 18.

## What paging changes, and what it doesn't

The math does not change at all. A row's slot `t` used to live at `K[row, t]`;
now it lives at

    pool[block_table[row, t // 16], :, t % 16]

and that is the whole of paging. Attention over those slots is the same
softmax over the same dot products. What changes is which bytes a decode step
has to touch to compute it.

That is why the first thing written was not the kernel but a reference in plain
PyTorch (`model/kernels/reference.py`): gather every block in the table into a
contiguous window, mask the tail, call the same SDPA the dense model calls. It
reads exactly the padding the kernel exists to avoid, so it is no faster than
day 7 -- but it is obviously correct, it runs on MPS so `make dev` keeps working
on a laptop, and it is the oracle every kernel test is checked against. The
tests build rows whose pages are shuffled across the pool, with spare blocks the
table must skip, and compare against plain per-row SDPA on contiguous K/V: if
the indirection is anything other than a relabelling, they fail.

## The pool

One pair of pools per layer, `[num_blocks + 1, 8 kv heads, 16 slots, 128 dims]`,
and one block table shared by all 36 layers -- a row's slot `t` is in the same
physical block in every layer, so the indirection is computed once per step.

Heads come before slots on purpose. One (block, head) is then a contiguous
16 x 128 x 2 B = 4 KB tile, which is what the kernel reads, and a warp can walk
it in straight 16-byte loads. Slots-before-heads would interleave the eight kv
heads and turn every load into a strided gather.

The pool holds 2,304 blocks: 18 rows x 128 blocks, the same 5.3 GB the dense
cache held. Today's change is about bytes *read*, not bytes *held*.

**Block 0 is the null page.** The allocator never hands it out, and every unused
table entry points at it. That is what makes the fixed launch shape safe: a
warmup step, a row with no request, a row past its last block -- all of them
write *somewhere*, and somewhere is a page nobody reads.

**The write is index math inside the graph.** `cache.write` maps positions to
slots on the device -- `block_table.gather(1, pos // 16)` and `pos % 16` -- so it
is captured with the rest of the step instead of being an index built on the host
per step. One indexing detail is easy to get wrong: in `pool[blk, :, off] = k`
the two advanced indices are separated by a slice, so PyTorch moves their
broadcast dimensions to the front and the destination is `[rows, T, heads, dim]`
-- exactly `k`'s shape, with no transpose.

**`move_row` stopped copying KV.** Retiring a row compacts the live ones down to
`0..n-1`. In the dense cache that moved the last row's whole prefix: 144 KB per
token across 36 layers, so ~58 MB for a 400-token row, on every retire. Now it
copies one 128-entry int32 row of the table, 512 bytes, and the KV stays where
it is. The test for it checks that the pools' bytes are untouched.

## Prefill does not need the kernel

A prompt starts at position 0 and there is no prefix sharing, so every key it
can attend to is one it is computing in the same forward. Prefill runs causal
SDPA on its own K and V directly and writes them into its pages for the decode
steps that follow; it never reads the pool. The paged kernel is decode-only,
which is also the only place it pays: prefill is compute-bound, decode is
bandwidth-bound.

The paged model reuses the dense one's norms, RoPE, MLP and parameter names, and
a test loads the same random weights into both and drives the engine's scenario
by hand -- three ragged prefills, 20 batched decode steps, retire the middle row,
compact, refill, 10 more steps -- asserting the logits agree at every step.

## The scheduler, three changes and no more

`model/inference_paged.py` is the continuous-batching engine with the cache
swapped underneath it.

- **admit** reserves a request's worst case, `ceil((prompt + 1024) / 16)` blocks,
  before its prefill. `submit` already refuses anything longer than a 2048-slot
  row, so each request needs at most 128 blocks, and with 18 rows and 2,304
  blocks the reservation *cannot fail*. No preemption, no eviction, nothing to
  get wrong. Allocating lazily and admitting more than 18 rows is the follow-up,
  and the `socrates_kv_blocks_free` gauge is there for it.
- **decode** has no read window. `seq_len` is derived on the device from the
  positions the step already feeds in -- the last one plus one -- so no new input
  crosses from the host and the launch shape depends only on the batch width.
- **retire** returns the row's blocks to the allocator, then compacts by copying
  one table row.

`Request`, the exceptions, the limits and the Prometheus counters are imported
from `inference_cont` rather than redefined: registering a counter name twice
raises at import, and the server catches `cont.QueueFull` whichever engine is
serving. `ENGINE=cont|paged` picks the engine; `/health` reports which, and the
benchmark records it, so a result file says what produced it.

The engine test pushes 24 requests through 18 rows in float64 and checks, at
every iteration of the loop, that no block is owned by two rows, that block 0
is never owned, and that owned plus free equals the pool. Then it runs every
request again alone, in a one-row engine, and requires identical tokens: batch
invariance, which is the property a scheduler bug breaks first.

## The kernel

`model/kernels/csrc/paged_attention.cu`, registered as
`torch.ops.socrates.paged_attention_decode`.

**One thread block per (kv head, row, partition), serving all four query heads
of the group.** This is the GQA saving made physical: each K and V byte is
loaded once and used four times. A kernel with a thread block per *query* head
would load the KV four times over and hand back most of what paging saved.

**Sixteen lanes per token.** A 128-dim bf16 head is 256 bytes, so 16 lanes with
one 16-byte load each cover a token, and a warp reads two tokens per load. Each
warp owns a 16-slot page at a time -- one contiguous 4 KB tile -- and issues all
eight of its loads before using any of them, so each thread has 128 bytes in
flight. Partial dot products for the four heads are reduced across the 16 lanes
with four `xor` shuffles, which never cross into the other half of the warp.

**Logits in shared memory, softmax per head, then V.** The query slice for all
four heads sits in registers for the whole K pass (32 floats a thread). Logits
go to shared memory, each head gets its max and sum by warp shuffles and a tiny
cross-warp reduce, and the V pass walks the same pages accumulating 4 heads x 8
dims in fp32 registers. Eight token groups are then summed through shared
memory and written out.

**Slots past a row's end are skipped, not multiplied by zero.** The tail of a
row's last page holds whatever that page's previous owner left there, and
`0 x NaN` is NaN. The CUDA test poisons every such slot, and the null page, with
`inf` and `NaN` and requires the output to be bit-identical to before.

### split-K, from the start

With one thread block per (kv head, row), the grid is 8 blocks at one row and
144 at eighteen, on 58 SMs. At low batch that leaves most of the card idle, and
a 2048-token row is walked by a single block. So `partition_size` P cuts each
row into `ceil(2048 / P)` partitions along the grid's z axis (flash-decoding).
Each partition writes an unnormalised partial output plus its max and sum, and
a second kernel combines them:

    M = max_p m_p        out = sum_p exp(m_p - M) o_p  /  sum_p exp(m_p - M) l_p

The grid is sized by the table width, which the host knows and which never
changes, while the work is sized by each row's real length, which only the
device knows: a partition that starts past its row's end exits on its first
instruction. That is how the launch shape stays fixed for CUDA graphs without
reading any padding. `partition_size=0` is the single-pass kernel, kept for the
comparison.

### fitting into torch.compile and CUDA graphs

The op is defined with `TORCH_LIBRARY` in `csrc/ops.cpp`, with no pybind11, so
`torch.ops.load_library` dlopens the `.so` and the registrations run. Its
schema marks no argument as mutated, which is true -- it reads the pools and
returns a fresh tensor -- so Inductor treats it as a pure function and the KV
writes in front of it stay ordinary in-place updates of module buffers. A
shape-only twin is registered in Python with `register_fake`, which is what
Dynamo traces through. The kernel launches on the current stream, reads no
value back to the host, and takes its split-K scratch from the caching
allocator, which inside a captured graph means the graph's private pool.

Two operational guards came from day 7's experience:

- `require_cuda_kernel()` runs in the engine's constructor. A missing or
  unloadable `.so` discovered on the first decode step would kill the serve
  thread while `/health` kept answering ok; discovered at startup it kills the
  process, and the container restarts loudly.
- `ATTN_IMPL=ref` routes CUDA through the PyTorch reference. If the paged engine
  ever misbehaves on the GPU, one run with it answers "kernel or paging?".

## Building it

`model/Dockerfile` is three stages now and the runtime image is still
`python:3.12-slim`:

    base     python, gcc, the locked dependencies (now including pytest)
    kernels  base + a slice of the CUDA 13.0 toolkit -> model/kernels/_C.abi3.so
    runtime  base + source + that one .so

The kernel stage is built `FROM base`, not from a CUDA image, so the extension
is compiled against the very torch, glibc and Python it is loaded into. The
toolkit is copied in from `nvidia/cuda:13.0.3-devel-ubuntu22.04` -- 13 because
cpp_extension refuses an nvcc whose major version differs from
`torch.version.cuda`, which is 13.0 for the locked torch 2.14.0 -- and only the
parts nvcc needs: the compiler binaries, NVVM, headers and `libcudart`. The full
devel tree is several GB of static math libraries, and day 7's disk filled
once already. It targets `sm_89` only.

The runtime stage loads the `.so` at build time. That needs no GPU -- it only
runs the op registrations -- so a kernel that cannot load (wrong torch, wrong
glibc, missing symbol) fails `docker build`, not the first request.

`infra/prod/kernel-on-instance.sh` builds the working tree on the GPU host as
`:dev` images, without pushing and without touching `:latest`, stops the
service, runs the CUDA tests, the kernel benchmark, the profiler or the
equivalence check in that image, and starts the service again.

Built natively on an arm64 Mac first (Docker Desktop, no GPU), which is enough to
prove everything except running the kernel: nvcc `13.0.88` compiles both files
in 23.7 s, the toolkit slice is **341 MB** instead of the devel image's several
GB, the `.so` is 3.8 MB, and the runtime stage prints

    kernel ok: socrates::paged_attention_decode(Tensor q, Tensor k_pool, Tensor v_pool,
               Tensor block_table, Tensor seq_lens, float scale, int partition_size) -> Tensor

The runtime image's layers are the old ones plus that one 3.8 MB file. Inside
it, `pytest` runs the 31 laptop tests and skips the 128 that need a GPU.

## Equivalence, first on the Mac

Before the kernel ran anywhere, the paged engine ran on MPS through the
reference attention, and the equivalence harness (`bench/day_8/equivalence.py`)
compared it with the dense engine on the 16 `PROMPTS`, all admitted together,
greedy, real checkpoint:

    cont (bucket 512) vs paged         13/16 identical
      how to make pizza?           diverge at token 28   top-2 gap 0.125 / 0.125
      explain recursion            diverge at token 77   top-2 gap 0.500 / 0.000
      how does a KV cache work?    diverge at token 44   top-2 gap 0.000 / 0.250

Three divergences, each at a step where the top two logits were tied or one
bf16 ulp apart (0.125 is one ulp for logits in [16, 32)). That looks like
rounding, but "looks like" is not evidence, so two controls:

    cont (bucket 512) vs cont (bucket 2048)    13/16 identical -- the same three
                                               prompts, at the same three tokens
    cont (bucket 2048) vs paged                16/16 identical

**The dense engine disagrees with itself in exactly the same places when only
its read window changes.** SDPA reduces over a masked window, and a 2048-slot
reduction rounds differently from a 512-slot one even though the extra slots
are masked. The PyTorch reference gathers all 128 table blocks -- a 2048-slot
window -- and matches the 2048-bucket dense engine token for token on all 16.
The divergences are a property of reduction width, not of paging.

Teacher forcing closes the remaining gap. The free-running comparison says
nothing after the first flip, because the two engines are then answering
different questions; feeding the paged engine the dense engine's tokens instead
of its own puts both on the same prefix at every step:

    teacher-forced, paged fed cont's tokens:   2,733 / 2,751 positions agree (99.35%)
                                               largest top-2 gap at a disagreement: 1 ulp

All 18 disagreements are at exact ties or one-ulp gaps. A paging bug -- a wrong
block, an off-by-one length, a stale slot read -- produces gaps of whole logits.

This matters for the gate set before starting: *the first 32 tokens identical on
all 16 prompts*. On the Mac that gate fails on "how to make pizza?" at token 28,
and the dense engine fails the same gate against itself. A literal first-N
identity gate can be failed by a correct implementation; what the gate is for --
"paging did not change the model" -- is what the controls above establish.

## On the L4

The spot instance had been reclaimed since day 7, so day 8's GPU work starts
with `terraform apply`: a fresh g6.2xlarge (30 minutes to win a spot slot), the
same 200 GB root volume, the stack pulled from ECR as before. On it,
`kernel-on-instance.sh` built the working tree as `:dev` -- 18 minutes, most of
it the first pull of the CUDA image and the torch layer -- and the root volume
went from 29% to 61%. That is the multi-stage build's cost on disk, and the
reason the script prunes before it builds.

The steps now run as a transient systemd unit on the host with an `EXIT` trap
that restarts the service, rather than inside the SSM command. The first
version ran them inline, and an SSM command that times out, or a laptop that
loses its SSO session halfway, would have left production down with nothing
left to bring it back. `compare-vllm.sh` arms a two-hour host-side timer for
the same reason.

### correctness

    pytest -m cuda      126 passed, 1 failed, 1 skipped (weights), 22.9 s

The 126 cover:

- the kernel against the fp32 oracle at every edge length, single pass and
  split-K at P = 128, 256, 512;
- 22 needle cases (an off-by-one at a page or partition seam moves the output
  from 64/n to 0, so it cannot hide in a tolerance) at all four partition sizes;
- GQA groups of 1, 2, 4 and 8;
- stale `inf`/`NaN` slots past a row's end, which must leave the output
  bit-identical;
- a raw `torch.cuda.graph` capture replayed over new lengths and new block
  tables;
- every bad-input error;
- `torch.compile(fullgraph=True)` through the fake.

The one failure was my test, not the kernel: it compared eager with compiled
logits elementwise, and one logit in 512 was 0.31 off. Both sides call the same
kernel; what differs is that eager rounds every bf16 intermediate (RMSNorm's
mean of squares, RoPE, SwiGLU) while Inductor's fused kernels keep them in
fp32 and round once, which comes to 1.5-1.8% per-row relative error, identical
on every row. The test now asks the question it was meant to ask: is the
compiled step far closer to this step's eager output than to the previous
step's, which is what a replay reading a stale length or table would produce.
Re-run on the L4 with that change:

    pytest -m cuda      127 passed, 1 skipped (weights), 18.5 s

    pytest -m weights   1 passed
      paged decode graphs warm for 1..18 rows in 282.4 s
      weights 8.04 GB  cache 5.44 GB  everything else 0.07 GB
      cudagraph_skips 0

The memory line is the check I was most worried about. Under
`torch.compile`, an in-place write to a module buffer is functionalised into
"compute a new tensor, copy it back", and if Inductor failed to turn that back
into an in-place write for the paged `index_put`, every captured graph would
own a private copy of the 5.4 GB pools. It did not: 70 MB is everything that is
neither weights nor cache.

### the kernel, alone

`bench/day_8/kernel_bench.py` times one attention layer both ways, inside CUDA
graphs, cycling through 8 layers' caches so the L2 is cold the way it is in a
real step. "Dense" is the dense model's own code: the bucketed window as a
strided view, the keep-mask, SDPA with `enable_gqa`. GB/s is the bytes each
design has to read -- the whole window for dense, live KV for paged.

    case          rows  mean len  window   dense SDPA            paged, one pass        speedup
    realistic      18      395     1024    649.7 us  116 GB/s    118.5 us  246 GB/s      5.48x
    B18_L390       18      390      512    331.6 us  114 GB/s    116.5 us  247 GB/s      2.85x
    B18_L1024      18     1024     1024    660.0 us  114 GB/s    292.6 us  258 GB/s      2.26x
    B18_L2048      18     2048     2048   1314.4 us  115 GB/s    579.2 us  261 GB/s      2.27x

"realistic" is day 7's regime: 18 rows averaging 395 tokens, the longest in
the 1024 bucket.

Two predictions, both wrong, in the useful direction:

- **I set the break-even at 16% of peak bandwidth. The kernel reaches 82%**
  (246 of 300 GB/s) on the realistic mix, and 87% on long rows. Sixteen
  lanes per token, eight 16-byte loads in flight per thread and one KV read for
  all four query heads is enough to stream pages at nearly the card's rate.
- **I assumed the dense attention ran at the step's 55% of peak. It runs at
  39%.** SDPA over a masked, strided window moves 116 GB/s regardless of
  shape. So attention was a bigger share of the dense step than the arithmetic
  allowed for: 23.4 ms of the 64.2 ms, not 16.5.

Look at the rows where the bytes are equal: at `B18_L2048` the window *is* the
live data, and paged is still 2.27x faster. A third of the win is not paging at
all. It is the kernel reading every KV byte once for four query heads, at 87%
of peak, where SDPA reads at 38%. Paging is the other two thirds: at
"realistic" the dense path reads 2.6x more bytes and the gap widens to 5.48x.

Per decode step at 18 rows, that is attention going from 23.4 ms to 4.3 ms --
**19 ms off a 64 ms step**, against the 10.6 ms the bandwidth arithmetic
predicted. If the rest of the step is unchanged, it lands near 45 ms.

### split-K, measured

    case        one pass   P=128    P=256    P=512
    B1_L390      18.7 us    9.8      13.1     19.8
    B1_L1024     37.6       10.4     13.2     21.9
    B1_L2048     85.8       37.1     36.4     37.2
    B4_L390      24.0       19.8     21.8     25.8
    B4_L1024     69.5       70.9     69.4     69.3
    B8_L1024    132.0      134.0    135.4    133.1
    B18_L1024   292.6      307.0    302.7    301.0
    realistic   118.5      121.2    120.4    121.4
    B1_L128      7.0        9.6       8.2      8.2

The occupancy argument holds exactly where it said it would and nowhere else.
One row is 8 thread blocks on 58 SMs, and splitting it is 2.3-3.6x faster.
From 8 rows (64 blocks) up, the single pass wins by 1-7%: the card is already
full, and the split only adds a reduce kernel and an fp32 round trip. Short
rows at low batch are the one case where splitting loses, by the cost of
launching the reduce.

So the partition is chosen per batch width rather than fixed:
`auto_partition` splits into 128-token partitions below 64 thread blocks
(fewer than 8 rows) and runs one pass above. The batch width is a static
shape under `dynamic=False`, so this is decided once per captured graph. It
only matters in the drain at the end of a run, but that is where day 7's
throughput curve spent its worst seconds.

A caveat on the low-batch rows: one row's KV for 8 layers is 32 MB at 1024
tokens, which fits in the L4's 48 MB L2. That is why `B1_L1024` at P=128
reports 405 GB/s, more than the card's DRAM bandwidth. The comparison between
partitions is fair, since every column is warm the same way, but the absolute
rates in those rows are L2 rates.

### the step, profiled

    ENGINE=paged profile_decode.py, 4 rows      35.15 ms / step,  0 cudagraph skips

The weights alone are 26.8 ms at 300 GB/s, so a 4-row step is now within 8.3 ms
of the weight-read floor, and moving 8.04 GB in 35.15 ms is 229 GB/s -- 76% of
peak for the whole step, against the 55% day 7 measured. Warmup is
282-324 s for 18 graphs, against 312 s for the dense engine's 72 in the same
session. Four times fewer graphs did not make warmup four times faster, for the
reason day 7 found: the first graph pays for compilation, and the rest reuse it.

### equivalence on CUDA

Compiled, CUDA graphs, all 16 prompts at once, as served:

    cont (512) vs paged              11/16 identical; diverge at tokens 5, 46, 48, 77, 102
    cont (512) vs cont (2048)        13/16 identical; diverge at tokens 141, 182, 189
    cont (2048) vs paged             11/16 identical; same five prompts, same five tokens

    teacher-forced, paged fed cont's tokens:
                                     2,892 / 2,917 positions agree (99.14%)
                                     largest top-2 gap at a disagreement: 1 bf16 ulp

**The 32-token gate fails, on one prompt.** "what is 5\*5?" diverges at token
5:

    cont:   5 multiplied by 5 equals 25.
    paged:  5 multiplied by 5 is **25**.

At that step the dense engine had "equals" ahead of "is" by 0.25, one bf16 ulp
at that magnitude; in the paged engine the two were *exactly* tied, and argmax
broke the tie by vocabulary index. Every divergence in every comparison is of
that kind: a top-two gap of 0 or 1 ulp on one side or the other.

To see what the model itself prefers at that step, the same prefix went through
the dense model eagerly on CPU, once in bf16 and once in fp32:

    fp32   logit[" equals"] 48.3811   logit[" is"] 48.1238   gap 0.2573
    bf16   logit[" equals"] 48.5000   logit[" is"] 48.2500   gap 0.25

The model prefers "equals" by 0.26 logits out of 48, which is exactly one bf16
ulp (bf16 values between 32 and 64 are spaced 0.25 apart). Any bf16
implementation that sums in a different order can round those two values onto
the same grid point. The paged kernel's order did; SDPA's did not. So the gate's
one failure comes from a near-tie in the model that bf16 cannot resolve, not
from anything paging introduced.

What changed from the Mac is the control. On MPS the dense engine split from
itself at the same tokens paged did; on CUDA it splits from itself later and
less often (3 prompts, all past token 140) than paged splits from either. That
is expected rather than alarming -- the dense engines share one SDPA kernel and
differ only in reduction width, while paged is a different algorithm with its
own summation order and its own `exp` -- but it means the control no longer
covers the gate on its own.

Teacher forcing covers what is left. When both engines see the same prefix,
they agree on 2,892 of 2,917 next-token choices, and the 25 disagreements are
all exact ties or one-ulp gaps. A wrong block, an off-by-one length or a stale
slot moves logits by whole units across a whole row; the needle tests rule that
out at the kernel level and this rules it out end to end. The literal criterion
-- identical first 32 tokens on all 16 -- is not met, and I have not changed it
to make it pass.

## The gate: cont, paged and vLLM, one session

`ENGINES="cont paged" TAG=dev ./infra/prod/compare-vllm.sh` runs the dense
engine, then the paged engine, then vLLM 0.29, back to back on the same box,
against the same 64 prompts on the same seed-0 Poisson arrivals. The baseline is
the dense engine from ten minutes earlier, not day 7's 200.4 tok/s from a
different instance.

    metric                    cont      paged      vllm   paged/cont   paged/vllm
    throughput avg tok/s     206.1      243.1     281.9     117.9%        86.2%
    throughput p50 tok/s     234.0      242.0     312.0     103.4%        77.6%
    throughput peak tok/s    342.0      396.0     454.0     115.8%        87.2%
    per-stream tok/s          15.8       21.5      26.4     136.3%        81.4%
    TTFT p95 s               22.68      11.76      7.79      51.8%       1.5x worse
    ITL p50 s               0.0634     0.0465    0.0379      73.4%       1.2x worse
    wall s                   107.1       91.0      79.3
    tokens made             22,064     22,113    22,359

    => gate: paged average throughput 243.1 vs cont 206.1 tok/s (+17.9%): PASS
       share of vLLM: cont 73.1% -> paged 86.2%

Token counts are within 0.2% of each other, so the ratio is speed, not workload.
The dense engine came in at 206.1 tok/s here against 200.4 on day 7's
instance -- a different box, a different spot slot, the same code -- which is
the reason the gate is a same-session comparison.

**Predicted: ~1.20x on ITL and close to +20% on saturated throughput. Measured:
1.36x on ITL and +17.9% on the run average.** The ITL beat the prediction
because the kernel beat its own: dense attention was 39% of peak, not 55%, so
there was more to remove. The run average came in under it for the reason
stated up front -- the ramp and the drain are not saturated. The saturated part
is the part to compare. Over the seconds where at least 17 rows were
generating:

    cont   273 tok/s   ->  18 rows / 273 = 65.9 ms a step
    paged  361 tok/s   ->  18 rows / 361 = 49.9 ms a step      (+32%)
    vllm   438 tok/s                                           paged = 82% of it

The kernel benchmark said attention would go from 23.4 ms a step to 4.3 ms; the
step went from 65.9 ms to 49.9, a 16 ms saving against 19 predicted. That is
close enough to say the saving is the attention, and that nothing else in the
step got worse.

Two numbers in the table need reading rather than quoting:

- **The median second barely moved (+3.4%).** Paged drains the queue faster, so
  more of its shorter run is ramp and drain, where the batch is half empty:
  mean rows generating fell from 12.2 to 11.3. The median second is therefore
  drawn from a different mix of phases in each run. The plateau above is the
  like-for-like figure.
- **TTFT p95 halved (22.7 s to 11.8 s)** with prefill untouched. As on day 7,
  TTFT is queue wait, and when arrivals exceed capacity the wait responds
  superlinearly: capacity rose from 18 / (371 tokens x 63.4 ms) = 0.77 req/s to
  1.04 req/s, still under the 2 req/s arrival rate.

## Switch-over

The gate passed, so `paged` is now the default -- in `docker-compose.yaml`, the
Makefile and `model/server.py` -- and production runs it:
`terraform apply` uploaded the compose file, `build-on-instance.sh` built and
pushed `:latest` and restarted the service, and

    https://socrates.pianofi.ca/health
      {"status":"ok","model":{"status":"ok","device":"cuda","max_batch":18,"engine":"paged"}}

    /api/chat "capital of France?"   The capital of France is **Paris**.   itl 37.8 ms

`ENGINE=cont` still selects the dense engine anywhere, and stays the baseline
for future comparisons.

## What is left

86% of vLLM's run-average throughput, 82% at the plateau, and 1.2x its ITL.

- **The step is no longer about attention.** At 18 rows attention is ~4.3 ms of
  a ~50 ms step. The rest is 8 GB of weights through cuBLAS and Inductor's
  kernels. Even a 4-row step (35.2 ms) is 8.3 ms over the 26.8 ms weight-read
  floor. vLLM's plateau step is 18 / 438 = 41 ms, so the remaining 9 ms is in
  the GEMMs and their fusion, not in anything paging can reach.
- **Sequential prefill.** TTFT p95 is still 1.5x vLLM's, and the queue is
  still the whole of it. Chunked, batched prefill is the next scheduler change.
- **The 3 host syncs per step** day 7 counted are untouched.

## The capacity follow-up

Today reserves every request's worst case at admission, so the pool is sized
like the dense cache it replaced -- 18 x 2048 slots -- and the 18-row cap
stands. The mean live context in this load is ~400 tokens against a 1,152-token
reservation (prompt + 1,024), so about two-thirds of the pool is reserved and
never written. Allocating blocks lazily, one block per 16 decode steps, would
let the same 5.4 GB hold roughly three times the rows. It also brings back the
problem the reservation avoided: a pool that can run dry mid-generation needs
preemption (evict a row, recompute it later). The `socrates_kv_blocks_free`
gauge is in place for that change, which is a different benchmark from this one
-- it has to be measured against vLLM unconstrained (`VLLM_MAX_SEQS` unset),
not against the 18-row yardstick.

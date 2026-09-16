Three changes today, in three commits: the benchmark now measures the streaming
path and reports throughput as a curve instead of one number; the same harness
now drives vLLM so our engine can be reported as a percentage of it; and the KV
cache moved into registered buffers so `mode="reduce-overhead"` can capture CUDA
graphs at all. The third one is the interesting write-up, because it worked
exactly as intended and bought nothing, and finding out why is the whole day.

Everything below was measured on the g6.2xlarge, one L4 (24 GB, ~300 GB/s),
Qwen3-4B in bf16, 18 rows of 2048 context, 64 prompts arriving Poisson at 2/s
with seed 0.

## The benchmark was not measuring the thing we serve

`/api/benchmark` drove `/generate`, the buffered endpoint. `/api/chat` streams.
So the only benchmark in the project never touched the code that actually
serves users -- the token queue hand-off from the decode thread, the delta
hold-back, the ndjson framing -- and any bug in it would have surfaced in the
browser rather than in a benchmark. It now drives `/stream`.

The second reason to stream is that a buffered call cannot tell you *when* its
tokens appeared. It hands back a total and a span, so the only throughput you
can compute is `sum(output_tokens) / wall_s`: one division for the whole run.
That number is not wrong, it is just an average over a shape it hides. Binning
tokens by their actual arrival time shows the shape:

    avg over the run   164 tok/s
    median second      189 tok/s
    peak second        216 tok/s

The average is 13% below a typical second because it includes the ramp, while
arrivals are still filling rows, and the drain at the end, where only the
longest requests are left and the batch is nearly empty. Plotting the row count
on the same axes makes the mechanism obvious: throughput tracks rows, not the
clock. 1.2 rows produced 30 tok/s; 18 rows produced 216.

A thing I got wrong first: I added a "throughput at full batch" figure, gated on
mean rows within a bin being >= MAX_BATCH - 0.5. It read 0.00 forever. Mean
occupancy inside a bin dips every time a row retires and refills, so with 1 s
bins it essentially never reaches 18 even when saturated -- the threshold was
measuring the bin width, not the engine. The median second answers the same
question and needs no threshold.

### a delta is not a token

To bin tokens by arrival time you have to count tokens, and the stream carries
*text*. Those do not correspond. The tokenizer renders an incomplete character
as U+FFFD, so `deltas()` holds the unstable tail back until the next token
completes it, which means one chunk can carry two tokens' worth of text and
another can carry none. Measured with the real tokenizer, "hi 🎉🚀 ok" is 6
tokens but 5 chunks: 🎉 spans tokens 3 and 4 and arrives as a single chunk. A
client counting lines would report 5 of 6 tokens, undercounting by 17%.

So `/stream` now emits `{"delta": text, "n": tokens_so_far}`, and `n` is taken
from the same snapshot of `r.output` that produced the text, so the two cannot
disagree. The live production stream shows it working:

    {"delta": " ", "n": 4}
    {"delta": "🎉", "n": 6}          <- one line, two tokens
    {"done": true, "output_tokens": 6, ...}

The invariant that matters is that the binned series sums to `output_tokens`
exactly. On the real run: 23,675 tokens, 23,675 in the bins.

## vLLM as the yardstick

The obvious way to compare against vLLM is to run vLLM's own benchmark script.
That would have been worthless: it defines throughput, TTFT and ITL its way and
we define them ours, so any difference between the two numbers would be partly
ours and partly a definition. Instead `/api/benchmark` takes `target=engine|vllm`
and speaks either our ndjson `/stream` or an OpenAI-compatible SSE endpoint. One
load generator, one set of arithmetic, two protocols.

Three request settings turned out to be load-bearing rather than cosmetic:

- **temperature 0.** Ours takes `argmax`. Sampling would change answer lengths,
  and answer lengths are the numerator of throughput.
- **max_tokens 1024**, matching MAX_NEW_TOKENS. Left alone vLLM keeps going, so
  it looks slower per request while doing strictly more work.
- **enable_thinking false.** Qwen3's chat template defaults to thinking mode.
  Ours passes `enable_thinking=False`; if vLLM does not, it spends hundreds of
  tokens reasoning before answering and the two runs are not the same workload
  at all. This is the one that would have quietly invalidated everything.

With all three, both engines emitted **23,792 tokens** on the first run and
23,675 vs 23,415 on the second (+1.1%) -- so the ratio is speed, not workload.
That is the check that makes the comparison mean anything, and it is worth
re-reading on every run.

ITL is now measured at the API tier for both. Taking ours from the engine's own
timing while vLLM's came off the wire would have flattered ours by exactly the
cost of a proxy hop. Where a chunk carries several tokens, its gap is divided
among them rather than counted once.

vLLM runs as a compose service behind a profile, so `up` never starts it, and
the comparison is sequential: an L4 has 24 GB, ours holds ~13 and vLLM is told
to take 85%, so running both at once would measure contention rather than
either engine. It gets the same 18-sequence and 2048-context caps we give
ourselves; unconstrained it would pick a much larger batch and win on capacity
we never gave ourselves. It shares the `weights` volume and `HF_HOME`, so it
reuses the 8 GB checkpoint instead of downloading it again.

    metric                    ours      vllm    ours/vllm
    throughput avg         164.1     291.1        56.4%
    throughput p50         189.5     319.5        59.3%
    throughput peak        216.0     449.0        48.1%
    per-stream              11.7      26.1        44.9%
    TTFT p95 (s)           51.89      9.40      5.5x worse
    ITL p50 (s)           0.0854    0.0383      2.2x worse
    wall (s)               144.2      80.4

**Ours reaches 56% of vLLM's aggregate throughput.**

Two things in that table are worth more than the headline. The peak ratio (48%)
is worse than the median ratio (59%), meaning vLLM pulls *further* ahead exactly
when the batch is full -- that points at the batched attention kernel rather
than at scheduling. And our occupancy is **higher** than vLLM's, 0.72 against
0.61, which is not a win: vLLM drains the queue so much faster that it spends
more of a shorter run with few rows live. Occupancy is a diagnostic, not a
score. The 5.5x TTFT gap is the most damning single number and it is structural:
our prefill is sequential, one request at a time, while vLLM chunks and batches
it, so under a queue our first-token latency degrades much faster.

## The KV cache, CUDA graphs, and a prediction that failed

Every decode step logged:

    skipping cudagraphs due to mutated inputs (72 instances)
    ... K[r, positions] = k

72 is 36 layers x (K and V): the entire cache. So `mode="reduce-overhead"` was
paying compilation and delivering none of the CUDA graphs it exists for.

The cause is ownership, not the write itself. `KVCache` was a plain Python
object passed into `forward()` as an argument, so Dynamo lifted its 72 tensors
into graph *inputs*, and `append()` mutated them. A CUDA graph replays a fixed
sequence of kernels against fixed addresses, so Inductor refuses to record a
graph that writes into an input -- the caller could pass different memory next
call and the replay would scribble on the wrong buffer. Parameters and buffers
are exempt, because they belong to the module and their addresses are stable.

So `KVCache` is now an `nn.Module` holding registered buffers, attached to the
model and reached through `self.cache` instead of being passed in. Two details
that are specific to this codebase:

- `persistent=False`. `load_state_dict` runs strict and no checkpoint contains a
  KV cache, so a persistent buffer fails the load on missing keys.
- The cache is built *after* the weights land, because the model is constructed
  on the meta device and a meta buffer cannot be copied to a real one.

There is an Inductor config, `cudagraph_support_input_mutation`, that silences
the check instead by copying the mutated inputs. That is the wrong trade here:
the cache is 144 KB/token x 2048 x 18 rows = 5.3 GB, so the copy would cost far
more than the launch overhead it saves.

`decode_step` also stopped allocating its two input tensors every step. They
were a host-to-device allocation on the critical path of the loop whose overhead
is the entire point, and CUDA graphs want to see storage that does not move.

**It worked, and it changed nothing.** The warning count went from firing on
every compiled width to zero. Throughput went from 164.0 to 164.1 tok/s. ITL p50
went from 85.0 ms to 85.4 ms. Our share of vLLM went from 57.2% to 56.4%, i.e.
noise. The predicted win -- the profiling that motivated this said 105 ms of
wall for 15 ms of GPU work -- did not appear.

The reason is that the bottleneck had already moved, and the arithmetic says so.
Count the bytes one decode step must read:

    weights                                              8.04 GB
    KV window: 18 rows x 36 layers x 2048 slots
               x 8 kv heads x 128 dim x 2 B x 2 (K,V)     5.44 GB
                                                        --------
                                                         13.48 GB

At the L4's ~300 GB/s that is a **45 ms floor per step**, and the profiler
measures the step at **80.9 ms** at 18 rows. So more than half the step is
irreducible memory traffic under the current design, and we are running at
roughly 55% of peak bandwidth. Launch overhead is no longer what is holding the
step back, so removing it was never going to show up.

And the reason we read 13.5 GB is a decision made earlier *to enable* compile
and CUDA graphs. `append()` returns the full `max_len` window rather than the
live prefix, because slicing to `int(positions.max()) + 1` needs that value on
the host -- a GPU-to-CPU sync per layer -- and it makes the returned shape grow
every step, so nothing traces once and graphs never form. The fixed window
solved that, and in solving it became the dominant cost. The optimisation that
made CUDA graphs possible is the one that made them pointless.

That also quantifies most of the gap to vLLM. Its per-step read is the same
8.04 GB of weights plus only the KV that is actually live -- a few hundred
tokens of context, tens of MB -- call it 8.1 GB against our 13.5 GB, so it moves
1.7x fewer bytes. Measured, its ITL is 2.2x better. Bytes account for most of
that, and kernel efficiency (it reaches ~70% of peak against our ~55%) for the
rest. This is what PagedAttention is *for*: reading live blocks instead of a
padded window.

So the next change is not a kernel and not a graph. It is making attention read
only live slots -- fixed-size blocks with a block table, so the shape stays
static without the window being 2048 wide. That is the change with 1.7x of
headroom behind it. Two smaller things are still on the table: the profiler
still reports 3 GPU syncs per decode step, and prefill is still sequential,
which is what the 5.5x TTFT gap is made of.

A note on the measurement itself: `overhead.py` reported `busy 185.6%`, which is
impossible and a sign the metric stopped being meaningful under CUDA graphs --
the profiler counts the graph replay and its constituent kernels both. The
`1.00x batch-1` column was also an artifact of measuring a single batch size.
The step time (80.9 ms) and the sync count are still good; the busy fraction is
not, and no conclusion above rests on it.

## Operational note

The 150 GB root volume filled during a deploy and took the box with it. Each
model build adds a 6.4 GB image that keeps its own git-sha tag, so it is never
dangling and never garbage-collected; three builds in a day was enough. The
failure mode is worth knowing because nothing about it says "disk": a full root
volume stops the SSM agent writing the script it was asked to run, so
`run-command` returns exit 1 with completely empty output, and cloud-init cannot
start either -- which means `growpart` cannot run, which means growing the EBS
volume does not help, because nothing can extend the filesystem into the new
space. The instance is only recoverable by replacement at that point.

`build-on-instance.sh` now prunes images older than 48h and caps the build cache
before building, prints `df` at each stage, and aborts instead of restarting the
service when a build or push returns non-zero. That last part mattered: the old
script reported "service start: Success" after a failed build, because the image
tags stayed on the previous build and the service came up happily on stale code.
The replacement instance has a 200 GB root volume and sits at 60% after a full
build.

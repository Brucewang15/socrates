"""The paged decode step under torch.compile(mode="reduce-overhead") on CUDA.

The point of keeping the cache in module buffers and the op free of host syncs
is that the whole step captures as a CUDA graph. These tests check that it
does -- zero cudagraph skips -- and that the captured step computes what the
eager one does.
"""

import pytest
import torch
from torch._dynamo.utils import counters

from model.paging import BlockAllocator, blocks_for
from model.qwen.qwen_kv_paged import PagedKVCache, Qwen3Paged
from tests.tiny import build, random_state

pytestmark = pytest.mark.cuda

# The kernel is compiled for head_dim 128 and 16-slot blocks, so the tiny model
# here is narrow everywhere except the head.
CUDA_TINY = {
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 128,
    "vocab_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
}
ROWS, MAX_LEN, BS = 4, 256, 16


def tiny_cuda():
    sd = random_state(Qwen3Paged, cfg=CUDA_TINY, dtype=torch.bfloat16)
    model = build(Qwen3Paged, sd, cfg=CUDA_TINY, dtype=torch.bfloat16, device="cuda")
    per_row = blocks_for(MAX_LEN, BS)
    cache = PagedKVCache(ROWS, 2, 2, 128, num_blocks=ROWS * per_row, block_size=BS,
                         max_blocks_per_row=per_row, dtype=torch.bfloat16, device="cuda")
    model.attach_cache(cache)
    return model, cache, BlockAllocator(ROWS * per_row)


@torch.no_grad()
def test_decode_step_captures_as_a_cuda_graph_and_matches_eager(monkeypatch):
    # Production splits rows into 256-slot partitions, and a 256-slot table here
    # would make that a single pass. 64 puts the split path -- scratch allocated
    # under capture, plus the reduce kernel -- inside the captured graph.
    from model import kernels
    monkeypatch.setattr(kernels, "PARTITION_SIZE", 64)
    torch._dynamo.reset()
    counters.clear()
    model, cache, alloc = tiny_cuda()
    g = torch.Generator().manual_seed(0)
    lens = [5, 40, 17, 100]
    for r, n in enumerate(lens):
        cache.assign(r, alloc.alloc(blocks_for(MAX_LEN, BS)))
        ids = torch.randint(1, 128, (1, n), generator=g).cuda()
        model(ids, slice(r, r + 1), torch.arange(n, device="cuda")[None])
        cache.lengths[r] = n

    compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
    step_ids = torch.zeros(ROWS, 1, dtype=torch.long, device="cuda")
    step_pos = torch.zeros(ROWS, 1, dtype=torch.long, device="cuda")
    prev = None
    for _ in range(6):                          # past warmup and recording into replay
        step_ids.copy_(torch.randint(1, 128, (ROWS, 1), generator=g))
        step_pos.copy_(torch.tensor(cache.lengths)[:, None])
        # eager first, then compiled: both write the same slot with the same k/v,
        # so the second write is a no-op up to rounding and both read one state
        want = model(step_ids, slice(0, ROWS), step_pos).float()
        got = compiled(step_ids, slice(0, ROWS), step_pos).float()
        # Both sides call the same kernel, so what differs is precision: eager
        # rounds every bf16 intermediate (RMSNorm's mean of squares, RoPE, SwiGLU)
        # while Inductor's fused kernels keep them in fp32 and round once at the
        # store. Measured on the L4: 1.5-1.8% per-row relative error, the same on
        # every row. What this test guards is the graph *following the data* --
        # a replay that read a stale length, position or block table computes a
        # different step altogether -- so it checks that the compiled output is
        # far closer to this step's eager output than to the previous step's.
        rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
        assert rel(got, want) < 0.05, f"compiled vs eager: {rel(got, want):.3f}"
        if prev is not None:
            assert rel(got, prev) > 4 * rel(got, want), (
                f"a different step is only {rel(got, prev):.3f} away; "
                f"the tolerance does not discriminate")
        prev = want
        for r in range(ROWS):
            cache.lengths[r] += 1

    assert counters["inductor"]["cudagraph_skips"] == 0, dict(counters["inductor"])


@pytest.mark.weights
def test_paged_engine_boots_compiled_and_serves():
    from model.inference_paged import PROMPTS, Engine
    torch._dynamo.reset()
    counters.clear()
    engine = Engine()                           # real weights, compiled, warmed
    # If Inductor failed to turn the functionalised KV write back into an
    # in-place one, every captured graph would hold its own copy of the pools --
    # 5.4 GB per copy -- and decode would pay for copying them every step.
    # Weights plus cache plus activations must be all there is.
    weights = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    cache = sum(b.numel() * b.element_size() for b in engine.cache.buffers())
    extra = torch.cuda.memory_allocated() - weights - cache
    print(f"weights {weights / 1e9:.2f} GB  cache {cache / 1e9:.2f} GB  "
          f"everything else {extra / 1e9:.2f} GB")
    assert extra < 2e9, f"{extra / 1e9:.2f} GB beyond weights and cache"
    reqs = [engine.submit(p) for p in PROMPTS[:4]]
    engine.admit()
    for _ in range(20):
        engine.sweep()
        if engine.n_active:
            engine.decode_step()
    engine.run()
    assert all(r.done for r in reqs)
    assert all(len(r.output) > 0 for r in reqs)
    assert engine.allocator.n_free == engine.allocator.num_blocks
    assert counters["inductor"]["cudagraph_skips"] == 0, dict(counters["inductor"])
    print({r.prompt: engine.tok.decode(r.output)[:80] for r in reqs})

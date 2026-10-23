"""The CUDA kernel against the PyTorch reference, on the GPU.

    uv run pytest -m cuda          # skipped unless CUDA and _C*.so are both present

The oracle is computed in fp32 from the same bf16 pools the kernel reads, so the
only differences left are the kernel's own: one bf16 rounding of the output
(half an ulp, 2^-9 relative) and fp32 accumulation order.

Random-data comparisons alone are weak at long lengths: with near-uniform
attention over 2048 tokens, dropping or adding one token moves the output by
~1/2048 of a value, which hides inside any sane tolerance. So the boundaries
are tested with needles instead -- q = 0 makes every weight exactly 1/n, V is
zero except at one slot, and the output is either 64/n or exactly 0 depending
on whether that slot was read. An off-by-one cannot hide in that.
"""

import random

import pytest
import torch

from model import kernels
from tests.test_reference import dense_oracle, make_case

pytestmark = pytest.mark.cuda

TOL = dict(atol=2e-3, rtol=1.6e-2)
EDGE = [1, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048]
PARTS = [0, 128, 256, 512]


def lens_from(seed, n, hi):
    rng = random.Random(seed)                   # one generator, n draws
    return [rng.randint(1, hi) for _ in range(n)]


def run(lens, partition, width=128, seed=0, **kw):
    q, kp, vp, bt, sl, dense = make_case(lens, width=width, dtype=torch.bfloat16,
                                         device="cuda", seed=seed, **kw)
    scale = q.shape[-1] ** -0.5
    got = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, scale, partition)
    want = dense_oracle(q, dense, scale)
    return got, want


@pytest.mark.parametrize("partition", PARTS)
@pytest.mark.parametrize("lens", [
    [1], [2048], EDGE, [2048, 1, 900, 17], lens_from(7, 18, 2048), lens_from(8, 4, 64),
], ids=["one", "full", "edges", "ragged4", "random18", "short4"])
def test_kernel_matches_oracle(lens, partition):
    got, want = run(lens, partition)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    torch.testing.assert_close(got.float(), want, **TOL)


@pytest.mark.parametrize("partition", [128, 256, 512])
def test_partition_boundaries(partition):
    P = partition
    lens = [P - 1, P, P + 1, 2 * P - 1, 2 * P, 2 * P + 1, 15, 16, 17]
    got, want = run(lens, P, seed=P)
    torch.testing.assert_close(got.float(), want, **TOL)


@pytest.mark.parametrize("hq,hkv", [(32, 8), (16, 16), (16, 8), (32, 4)])
def test_gqa_groups(hq, hkv):
    got, want = run([300, 5, 1500], 256, hq=hq, hkv=hkv)
    torch.testing.assert_close(got.float(), want, **TOL)


def needle_case(n, slot, hq=32, hkv=8, d=128, bs=16):
    """One row of length n, q = 0, V zero except V[slot] = 8 * (kv head + 1).

    The table covers slot even when slot >= n, so a kernel that reads one
    slot too far finds the needle instead of the null page.
    """
    nb = -(-max(n, slot + 1) // bs)
    blocks = torch.randperm(nb + 5)[:nb] + 1                  # shuffled, never block 0
    kp = torch.randn(nb + 6, hkv, bs, d).to(torch.bfloat16)
    vp = torch.zeros(nb + 6, hkv, bs, d, dtype=torch.bfloat16)
    for h in range(hkv):
        vp[blocks[slot // bs], h, slot % bs] = 8.0 * (h + 1)
    table = torch.zeros(1, 128, dtype=torch.int32)
    table[0, :nb] = blocks.to(torch.int32)
    q = torch.zeros(1, hq, d, dtype=torch.bfloat16)
    sl = torch.tensor([n], dtype=torch.int32)
    # expected: query head i reads kv head i // G; weight 1/n on the needle if read
    G = hq // hkv
    head_val = torch.tensor([8.0 * (i // G + 1) for i in range(hq)])
    want = (head_val / n if slot < n else torch.zeros(hq))[None, :, None].expand(1, hq, d)
    cuda = lambda t: t.cuda()
    return cuda(q), cuda(kp), cuda(vp), cuda(table), cuda(sl), want.cuda()


@pytest.mark.parametrize("partition", PARTS)
@pytest.mark.parametrize("n,slot", [
    # the last live slot must be read ...
    (1, 0), (16, 15), (17, 16), (256, 255), (257, 256), (1024, 1023), (2048, 2047),
    # ... and the first slot past the end must not be
    (1, 1), (15, 15), (16, 16), (255, 255), (256, 256), (1024, 1024), (2047, 2047),
    # partition seams, read from either side
    (600, 127), (600, 128), (600, 255), (600, 256), (1100, 511), (1100, 512),
    (2048, 0), (2048, 1536),
])
def test_needles(n, slot, partition):
    q, kp, vp, bt, sl, want = needle_case(n, slot)
    got = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, 0.1, partition)
    torch.testing.assert_close(got.float(), want, atol=1e-4, rtol=8e-3)


def test_split_and_single_pass_agree():
    lens = lens_from(9, 18, 2048)
    q, kp, vp, bt, sl, _ = make_case(lens, width=128, dtype=torch.bfloat16, device="cuda")
    s = q.shape[-1] ** -0.5
    one = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, 0)
    for P in (128, 256, 512):
        split = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, P)
        torch.testing.assert_close(split.float(), one.float(), **TOL)


@pytest.mark.parametrize("partition", [0, 256])
def test_stale_slots_and_garbage_past_seq_len_are_ignored(partition):
    q, kp, vp, bt, sl, _ = make_case([20, 7, 300, 1025, 2047], width=128,
                                     dtype=torch.bfloat16, device="cuda")
    s = q.shape[-1] ** -0.5
    before = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, partition)
    # poison every slot past each row's length in its last page
    for b, n in enumerate(sl.tolist()):
        last = int(bt[b, (n - 1) // 16])
        kp[last, :, n % 16 or 16:] = float("inf")
        vp[last, :, n % 16 or 16:] = float("nan")
    after = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, partition)
    assert torch.isfinite(after).all()
    torch.testing.assert_close(after, before)


def test_replays_in_a_cuda_graph_with_new_lengths_and_tables():
    """The production property: shapes fixed at capture, data changed in place.

    Capture once, then rewrite seq_lens and the block table in their own storage
    and replay. The replay must follow the new data, which is what lets one
    graph per batch width serve every context length.
    """
    # Captured at the largest pool any replay below needs, so none is skipped.
    q, kp, vp, bt, sl, _ = make_case([2048] * 4, width=128, dtype=torch.bfloat16,
                                     device="cuda")
    s = q.shape[-1] ** -0.5
    op = lambda: torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, 256)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op()

    for i, new in enumerate(([1, 2047, 17, 900], [300, 40, 2000, 5], [16, 15, 257, 1])):
        q2, kp2, vp2, bt2, sl2, _ = make_case(new, width=128, dtype=torch.bfloat16,
                                              device="cuda", seed=i + 1)
        assert kp2.shape[0] <= kp.shape[0]
        kp[:kp2.shape[0]].copy_(kp2)
        vp[:vp2.shape[0]].copy_(vp2)
        q.copy_(q2)
        bt.copy_(bt2)
        sl.copy_(sl2)
        graph.replay()
        torch.cuda.synchronize()
        want = torch.ops.socrates.paged_attention_decode(q, kp, vp, bt, sl, s, 256)
        torch.testing.assert_close(out, want)


def test_matches_the_reference_path_through_the_dispatcher():
    q, kp, vp, bt, sl, _ = make_case([33, 700, 2048], width=128, dtype=torch.bfloat16,
                                     device="cuda")
    s = q.shape[-1] ** -0.5
    got = kernels.paged_attention_decode(q, kp, vp, bt, sl, s)
    ref = kernels.paged_attention_decode_ref(q.float(), kp.float(), vp.float(), bt, sl, s)
    torch.testing.assert_close(got.float(), ref, **TOL)


def test_bad_inputs_raise():
    q, kp, vp, bt, sl, _ = make_case([5], width=4, dtype=torch.bfloat16, device="cuda")
    op = torch.ops.socrates.paged_attention_decode
    with pytest.raises(RuntimeError, match="bfloat16"):
        op(q.float(), kp, vp, bt, sl, 0.1, 0)
    with pytest.raises(RuntimeError, match="int32"):
        op(q, kp, vp, bt.long(), sl, 0.1, 0)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(q, kp.transpose(1, 2), vp.transpose(1, 2), bt, sl, 0.1, 0)
    with pytest.raises(RuntimeError, match="multiple of 16"):
        op(q, kp, vp, torch.zeros(1, 128, dtype=torch.int32, device="cuda"), sl, 0.1, 100)
    with pytest.raises(RuntimeError, match="CUDA"):
        op(q.cpu(), kp, vp, bt, sl, 0.1, 0)
    with pytest.raises(RuntimeError, match="not compiled"):
        q3, kp3, vp3, bt3, sl3, _ = make_case([5], hq=9, hkv=3, width=4,
                                              dtype=torch.bfloat16, device="cuda")
        op(q3, kp3, vp3, bt3, sl3, 0.1, 0)


def test_traces_under_torch_compile():
    q, kp, vp, bt, sl, _ = make_case([40, 3], width=8, dtype=torch.bfloat16, device="cuda")
    s = q.shape[-1] ** -0.5

    def f(q, kp, vp, bt, sl):
        return kernels.paged_attention_decode(q * 1.0, kp, vp, bt, sl, s, 64) + 0.0

    eager = f(q, kp, vp, bt, sl)
    compiled = torch.compile(f, fullgraph=True)(q, kp, vp, bt, sl)
    torch.testing.assert_close(compiled, eager)

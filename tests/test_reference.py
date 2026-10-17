"""The paged reference against plain per-row SDPA on contiguous K/V.

Shared with test_kernel_cuda.py: `make_case` builds a batch whose rows live in
shuffled, non-contiguous pages, alongside the contiguous K/V each row would have
had in the dense cache. Agreement means the indirection is exactly a relabelling.
"""

import random

import pytest
import torch
from torch.nn import functional as F

from model.kernels.reference import paged_attention_decode_ref


def make_case(lens, hq=32, hkv=8, d=128, bs=16, width=None, dtype=torch.float32,
              device="cpu", seed=0):
    """Random queries, and K/V scattered into shuffled pages of a shared pool.

    Returns (q, k_pool, v_pool, block_table, seq_lens, dense) where dense[b] is
    row b's contiguous (k, v), each [hkv, len, d], for the oracle.
    """
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    B = len(lens)
    width = width or max(-(-n // bs) for n in lens)
    need = sum(-(-n // bs) for n in lens)
    num_blocks = need + 7                       # spare blocks: tables must skip them
    physical = list(range(1, num_blocks + 1))   # block 0 stays the null page
    rng.shuffle(physical)

    k_pool = torch.randn(num_blocks + 1, hkv, bs, d, generator=g)
    v_pool = torch.randn(num_blocks + 1, hkv, bs, d, generator=g)
    q = torch.randn(B, hq, d, generator=g)
    table = torch.zeros(B, width, dtype=torch.int32)
    dense = []
    for b, n in enumerate(lens):
        nb = -(-n // bs)
        blocks = [physical.pop() for _ in range(nb)]
        table[b, :nb] = torch.tensor(blocks, dtype=torch.int32)
        # contiguous view of exactly what this row wrote: slot t at block t//bs
        k = k_pool[blocks].permute(1, 0, 2, 3).reshape(hkv, nb * bs, d)[:, :n]
        v = v_pool[blocks].permute(1, 0, 2, 3).reshape(hkv, nb * bs, d)[:, :n]
        dense.append((k, v))

    seq_lens = torch.tensor(lens, dtype=torch.int32)
    cast = lambda t: t.to(device=device, dtype=dtype)
    return (cast(q), cast(k_pool), cast(v_pool), table.to(device), seq_lens.to(device),
            [(cast(k), cast(v)) for k, v in dense])


def dense_oracle(q, dense, scale):
    """Row by row, each against its own contiguous K/V, no masking at all."""
    outs = []
    for b, (k, v) in enumerate(dense):
        o = F.scaled_dot_product_attention(
            q[b][None, :, None, :].float(), k[None].float(), v[None].float(),
            scale=scale, enable_gqa=True)
        outs.append(o[0, :, 0, :])
    return torch.stack(outs)


EDGE = [1, 15, 16, 17, 31, 32, 33, 255, 256, 257]
# One generator, many draws. random.Random(3).randint(...) inside a comprehension
# re-seeds on every element and yields the same length eighteen times.
_rng = random.Random(3)
RAGGED18 = [_rng.randint(1, 2048) for _ in range(18)]


@pytest.mark.parametrize("lens", [
    [1], [16], [17], [2048], EDGE, [2048, 1, 900, 17], RAGGED18,
])
def test_reference_matches_dense(lens):
    q, kp, vp, bt, sl, dense = make_case(lens)
    scale = q.shape[-1] ** -0.5
    got = paged_attention_decode_ref(q, kp, vp, bt, sl, scale)
    torch.testing.assert_close(got, dense_oracle(q, dense, scale), atol=1e-5, rtol=1e-5)


def test_reference_small_gqa_and_wider_table():
    # a different group ratio, and table columns past every row's last block
    q, kp, vp, bt, sl, dense = make_case([5, 40, 3], hq=6, hkv=2, d=32, bs=8, width=12)
    scale = 0.3
    got = paged_attention_decode_ref(q, kp, vp, bt, sl, scale)
    torch.testing.assert_close(got, dense_oracle(q, dense, scale), atol=1e-5, rtol=1e-5)


def test_stale_pages_past_seq_len_are_ignored():
    q, kp, vp, bt, sl, dense = make_case([20, 7])
    scale = q.shape[-1] ** -0.5
    before = paged_attention_decode_ref(q, kp, vp, bt, sl, scale)
    # scribble over every slot past each row's length, and the null page
    for b, n in enumerate(sl.tolist()):
        for t in range(n, bt.shape[1] * kp.shape[2]):
            blk = int(bt[b, t // kp.shape[2]])
            kp[blk, :, t % kp.shape[2]] = 1e4
            vp[blk, :, t % kp.shape[2]] = -1e4
    after = paged_attention_decode_ref(q, kp, vp, bt, sl, scale)
    torch.testing.assert_close(after, before)


def test_auto_partition_splits_only_small_batches():
    from model.kernels import auto_partition
    assert auto_partition(1, 8) == 128          # 8 thread blocks on 58 SMs: split
    assert auto_partition(7, 8) == 128
    assert auto_partition(8, 8) == 0            # 64 blocks: one pass wins
    assert auto_partition(18, 8) == 0

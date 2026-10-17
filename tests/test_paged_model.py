"""Qwen3Paged against qwen_kv_cont.Qwen3 with identical weights, step for step.

The scenario is the engine's, driven by hand: three ragged prefills, twenty
batched decode steps, retire the middle row and compact, refill the freed row,
and keep decoding. Every logit at every step has to agree.
"""

import torch

from model.paging import BlockAllocator, blocks_for
from model.qwen.qwen_kv_cont import KVCache, Qwen3
from model.qwen.qwen_kv_paged import PagedKVCache, Qwen3Paged
from tests.tiny import TINY, build, random_state

MAX_LEN, BS, ROWS = 64, 8, 3


def models():
    sd = random_state(Qwen3)
    dense, paged = build(Qwen3, sd), build(Qwen3Paged, sd)
    hkv, hd, L = TINY["num_key_value_heads"], TINY["head_dim"], TINY["num_hidden_layers"]
    dense.attach_cache(KVCache(ROWS, L, hkv, hd, max_len=MAX_LEN, dtype=torch.float32,
                               device="cpu"))
    nb = ROWS * MAX_LEN // BS
    paged.attach_cache(PagedKVCache(ROWS, L, hkv, hd, num_blocks=nb, block_size=BS,
                                    max_blocks_per_row=MAX_LEN // BS,
                                    dtype=torch.float32, device="cpu"))
    return dense, paged, BlockAllocator(nb)


def test_parameter_names_match_the_dense_model():
    assert set(Qwen3(TINY).state_dict()) == set(Qwen3Paged(TINY).state_dict())


@torch.no_grad()
def test_paged_matches_dense_through_retire_and_refill():
    dense, paged, alloc = models()
    g = torch.Generator().manual_seed(1)
    rows = {}            # row -> list of tokens so far (prompt + generated)

    def prefill(row, n):
        ids = torch.randint(1, TINY["vocab_size"], (1, n), generator=g)
        pos = torch.arange(n)[None]
        paged.cache.assign(row, alloc.alloc(blocks_for(MAX_LEN, BS)))
        a = dense(ids, slice(row, row + 1), pos, n)
        b = paged(ids, slice(row, row + 1), pos)
        torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)
        dense.cache.lengths[row] = paged.cache.lengths[row] = n
        rows[row] = ids[0].tolist() + [int(a[0, -1].argmax())]

    def decode(steps):
        for _ in range(steps):
            n = len(rows)
            ids = torch.tensor([[rows[r][-1]] for r in range(n)])
            pos = torch.tensor([[dense.cache.lengths[r]] for r in range(n)])
            a = dense(ids, slice(0, n), pos, MAX_LEN)
            b = paged(ids, slice(0, n), pos)
            torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)
            for r in range(n):
                dense.cache.lengths[r] += 1
                paged.cache.lengths[r] += 1
                rows[r].append(int(a[r, -1].argmax()))

    for row, n in enumerate([5, 17, 9]):         # 17 crosses two block boundaries
        prefill(row, n)
    decode(20)

    # retire row 1: free its blocks, compact row 2 down into it
    alloc.free(paged.cache.release(1))
    paged.cache.move_row(2, 1)
    dense.cache.move_row(2, 1)
    rows[1] = rows.pop(2)

    prefill(2, 11)                               # refill the freed row
    decode(10)
    assert alloc.n_free == alloc.num_blocks - 3 * blocks_for(MAX_LEN, BS)

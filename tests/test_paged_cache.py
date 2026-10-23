import torch

from model.kernels.reference import paged_attention_decode_ref
from model.paging import BlockAllocator, blocks_for
from model.qwen.qwen_kv_paged import PagedKVCache

HKV, D, BS = 2, 8, 4


def cache(max_batch=3, num_blocks=24, width=6):
    return PagedKVCache(max_batch, n_layers=2, n_kv_heads=HKV, head_dim=D,
                        num_blocks=num_blocks, block_size=BS, max_blocks_per_row=width,
                        dtype=torch.float32, device="cpu")


def gather_row(c, layer, row, n):
    """Read a row's first n slots back through the table, [n, HKV, D] each."""
    bt = c.block_table[row].long()
    k = c.layers[layer].k[bt].permute(0, 2, 1, 3).reshape(-1, HKV, D)[:n]
    v = c.layers[layer].v[bt].permute(0, 2, 1, 3).reshape(-1, HKV, D)[:n]
    return k, v


def test_ragged_writes_read_back_through_the_table():
    c, alloc = cache(), BlockAllocator(24)
    lens = [3, 9, 6]
    # interleave allocations so no row's blocks are contiguous
    owned = [[] for _ in lens]
    for i in range(max(blocks_for(n, BS) for n in lens)):
        for r, n in enumerate(lens):
            if i < blocks_for(n, BS):
                owned[r] += alloc.alloc(1)
    for r in range(3):
        c.assign(r, owned[r])

    truth = {r: (torch.randn(n, HKV, D), torch.randn(n, HKV, D)) for r, n in enumerate(lens)}
    # prefill each row on its own, as the engine does
    for r, n in enumerate(lens):
        k, v = truth[r]
        c.write(1, slice(r, r + 1), k[None], v[None], torch.arange(n)[None])
    for r, n in enumerate(lens):
        k, v = gather_row(c, 1, r, n)
        torch.testing.assert_close(k, truth[r][0])
        torch.testing.assert_close(v, truth[r][1])

    # one batched decode write at each row's own position
    k1, v1 = torch.randn(3, 1, HKV, D), torch.randn(3, 1, HKV, D)
    c.write(1, slice(0, 3), k1, v1, torch.tensor(lens)[:, None])
    for r, n in enumerate(lens):
        k, v = gather_row(c, 1, r, n + 1)
        torch.testing.assert_close(k[n], k1[r, 0])
        torch.testing.assert_close(v[:n], truth[r][1])
    # layer 0 was never written
    assert c.layers[0].k.abs().sum() == 0


def test_move_row_copies_the_table_not_the_kv():
    c = cache()
    c.assign(0, [5, 6])
    c.assign(2, [9, 1, 3])
    n = 10
    k, v = torch.randn(n, HKV, D), torch.randn(n, HKV, D)
    c.write(0, slice(2, 3), k[None], v[None], torch.arange(n)[None])
    c.lengths[2] = n

    ptr = c.layers[0].k.data_ptr()
    before = c.layers[0].k.clone()
    c.release(0)
    c.move_row(2, 0)

    assert c.layers[0].k.data_ptr() == ptr
    assert torch.equal(c.layers[0].k, before)            # not one byte of KV moved
    assert c.row_blocks[0] == [9, 1, 3] and c.row_blocks[2] == []
    assert c.lengths == [n, 0, 0]
    assert c.block_table[2].abs().sum() == 0
    got_k, got_v = gather_row(c, 0, 0, n)
    torch.testing.assert_close(got_k, k)
    torch.testing.assert_close(got_v, v)


def test_released_and_unassigned_rows_write_only_the_null_page():
    c = cache()
    c.assign(1, [4])
    c.release(1)
    k, v = torch.ones(3, 1, HKV, D), torch.ones(3, 1, HKV, D)
    c.write(0, slice(0, 3), k, v, torch.tensor([[0], [1], [2]]))
    pool = c.layers[0].k
    assert pool[1:].abs().sum() == 0                     # every real block untouched
    assert pool[0].abs().sum() > 0


def test_cache_feeds_the_reference_kernel():
    c = cache(max_batch=2)
    c.assign(0, [7, 2])
    c.assign(1, [11])
    q = torch.randn(2, 4, D)
    for r, n in [(0, 6), (1, 3)]:
        c.write(0, slice(r, r + 1), torch.randn(1, n, HKV, D), torch.randn(1, n, HKV, D),
                torch.arange(n)[None])
    out = paged_attention_decode_ref(q, c.layers[0].k, c.layers[0].v, c.block_table,
                                     torch.tensor([6, 3], dtype=torch.int32), D ** -0.5)
    for r, n in [(0, 6), (1, 3)]:
        k, v = gather_row(c, 0, r, n)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[r][None, :, None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
            enable_gqa=True)[0, :, 0]
        torch.testing.assert_close(out[r], ref)

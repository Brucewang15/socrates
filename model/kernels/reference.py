"""Paged decode attention in plain PyTorch: the oracle, and the MPS/CPU path.

This is not the fast path and does not try to be. It gathers every block in the
table -- live or not -- into a contiguous window and masks the tail, which is
exactly the padded read the CUDA kernel exists to avoid. What it is good for is
being obviously correct: the indirection through the block table is one gather,
and everything after that is the same SDPA call the dense model makes.
"""

import torch
from torch.nn import functional as F


def paged_attention_decode_ref(q, k_pool, v_pool, block_table, seq_lens, scale: float):
    """One query token per row against that row's own pages.

    q:           [B, Hq, D]              this step's queries, post-RoPE
    k_pool:      [num_blocks, Hkv, BS, D]
    v_pool:      [num_blocks, Hkv, BS, D]
    block_table: [B, W] int32            physical block of each row's logical block
    seq_lens:    [B] int32               slots each row has written, this step's included
    returns      [B, Hq, D]

    No host sync: seq_lens only ever appears inside a comparison, so it never has
    to come back to the CPU.
    """
    B, Hq, D = q.shape
    _, Hkv, BS, _ = k_pool.shape
    W = block_table.shape[1]
    bt = block_table.long()

    # [B, W, Hkv, BS, D] -> [B, Hkv, W*BS, D]: logical slot t of row b sits at
    # block bt[b, t // BS], offset t % BS. That is the whole of paging.
    k = k_pool[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, W * BS, D)
    v = v_pool[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, W * BS, D)

    slot = torch.arange(W * BS, device=q.device)
    keep = (slot[None, :] < seq_lens[:, None].to(slot.dtype))[:, None, None, :]   # [B,1,1,S]

    out = F.scaled_dot_product_attention(
        q[:, :, None, :], k, v, attn_mask=keep, scale=scale, enable_gqa=True,
    )                                                                          # [B,Hq,1,D]
    return out[:, :, 0, :]

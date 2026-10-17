"""One entry point for paged decode attention, whatever the device.

    CUDA, extension built   -> torch.ops.socrates.paged_attention_decode
    anything else           -> the pure-PyTorch reference
    ATTN_IMPL=ref           -> the reference, even on CUDA

ATTN_IMPL=ref is for bisecting: if the paged engine misbehaves on the GPU, run it
again with the reference and the answer is either "the kernel" or "the paging".

The extension is loaded with torch.ops.load_library rather than imported as a
Python module, so the .so needs no Python bindings at all -- only the
TORCH_LIBRARY static initialisers, which run on load and register the op.
"""

import glob
import os
from pathlib import Path

import torch

from model.kernels.reference import paged_attention_decode_ref

ATTN_IMPL = os.getenv("ATTN_IMPL", "auto")
if ATTN_IMPL not in ("auto", "cuda", "ref"):
    raise ValueError(f"ATTN_IMPL must be auto, cuda or ref, not {ATTN_IMPL!r}")

# Tokens per thread block in the split-K kernel; 0 means one pass over the whole
# row. "auto" picks per batch width, from bench/day_8/kernel_bench.py on the L4
# (docs/day_8_docs.md): splitting only pays while (rows x kv heads) thread blocks
# leave the 58 SMs idle. At 18 rows the single pass is 1-7% faster than any
# split; at one row a 128-token split is 1.9-3.6x faster than the single pass.
# The batch width is a static shape under torch.compile(dynamic=False), so this
# is decided once per captured graph, not per step.
_PARTITION_ENV = os.getenv("PAGED_PARTITION", "auto")
PARTITION_SIZE: int | None = None if _PARTITION_ENV == "auto" else int(_PARTITION_ENV)
SPLIT_BELOW_BLOCKS = 64        # rows x kv heads under which the grid is split
SPLIT_PARTITION = 128


def auto_partition(rows: int, kv_heads: int) -> int:
    return SPLIT_PARTITION if rows * kv_heads < SPLIT_BELOW_BLOCKS else 0

_HERE = Path(__file__).resolve().parent
_loaded = False
_load_error: str | None = None


def _load() -> None:
    global _loaded, _load_error
    found = sorted(glob.glob(str(_HERE / "_C*.so")))
    if not found:
        _load_error = f"no _C*.so in {_HERE}; build it with model/kernels/setup.py"
        return
    try:
        torch.ops.load_library(found[0])
    except OSError as e:                      # wrong CUDA, wrong torch, wrong glibc
        _load_error = f"{found[0]}: {e}"
        return

    # Shape-only twin, so Dynamo can trace through the op without running it.
    # Output is a fresh contiguous [B, Hq, D] in q's dtype -- contiguous because
    # that is what the CUDA side returns, and Inductor trusts these strides.
    # Nothing is mutated, which is what lets the KV writes before it stay
    # in-place under Inductor.
    @torch.library.register_fake("socrates::paged_attention_decode")
    def _(q, k_pool, v_pool, block_table, seq_lens, scale, partition_size):
        torch._check(q.dim() == 3)
        torch._check(block_table.dim() == 2 and block_table.shape[0] == q.shape[0])
        return q.new_empty(q.shape)

    _loaded = True


if torch.cuda.is_available():
    _load()


def cuda_kernel_loaded() -> bool:
    return _loaded


def require_cuda_kernel() -> None:
    """Fail at startup, not on the first decode step.

    A missing kernel discovered inside the serve thread kills that thread while
    /health keeps answering ok -- the failure mode day 7's Dockerfile note
    describes for a missing compiler. Raising here takes the process down instead.
    """
    if ATTN_IMPL == "ref":
        return
    if not _loaded:
        raise RuntimeError(f"paged engine on CUDA needs the socrates kernel: {_load_error}")


def paged_attention_decode(q, k_pool, v_pool, block_table, seq_lens, scale: float,
                           partition_size: int | None = None):
    """[B, Hq, D] queries against each row's own pages; see reference.py for shapes."""
    if q.is_cuda and ATTN_IMPL != "ref":
        if not _loaded:
            raise RuntimeError(f"socrates kernel unavailable: {_load_error}")
        if partition_size is None:
            partition_size = (PARTITION_SIZE if PARTITION_SIZE is not None
                              else auto_partition(q.shape[0], k_pool.shape[1]))
        return torch.ops.socrates.paged_attention_decode(
            q, k_pool, v_pool, block_table, seq_lens, scale, partition_size)
    return paged_attention_decode_ref(q, k_pool, v_pool, block_table, seq_lens, scale)

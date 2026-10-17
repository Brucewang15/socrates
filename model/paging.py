"""Host-side bookkeeping for the paged KV cache: which physical blocks are free.

Nothing here touches the device. The scheduler already knows every row's length,
so it can also know which blocks each row owns, and keeping that on the host is
what lets a decode step run without asking the GPU anything.

Block 0 is never handed out. It is the null page: every unused block-table
entry points at it, so a row with no blocks (warmup, a retired row, a row past
its last block) still writes somewhere harmless instead of somewhere random.
"""


class OutOfBlocks(Exception):
    pass


def blocks_for(tokens: int, block_size: int) -> int:
    """How many blocks `tokens` slots occupy, rounded up."""
    return -(-tokens // block_size)


class BlockAllocator:
    """A free list over physical blocks 1..num_blocks; block 0 is reserved.

    alloc/free are O(n) in the blocks moved and never touch the device. Freed
    blocks go back on top of the stack, so a refill reuses what was just
    released -- warm in L2, for whatever that is worth at 4 KB a tile.
    """

    NULL = 0

    def __init__(self, num_blocks: int):
        if num_blocks < 1:
            raise ValueError("need at least one usable block")
        self.num_blocks = num_blocks
        # popped from the end, so the first allocation gets 1, 2, 3, ...
        self._free = list(range(num_blocks, 0, -1))
        self._used: set[int] = set()

    @property
    def n_free(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks")
        if n > len(self._free):
            raise OutOfBlocks(f"asked for {n} blocks, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        self._used.update(out)
        return out

    def free(self, blocks: list[int]) -> None:
        # Validate everything before changing anything, so a bad call leaves the
        # allocator exactly as it was.
        seen = set()
        for b in blocks:
            if b == self.NULL:
                raise ValueError("block 0 is the null page and is never allocated")
            if b not in self._used or b in seen:
                raise ValueError(f"block {b} freed twice, or never allocated")
            seen.add(b)
        for b in reversed(blocks):
            self._used.remove(b)
            self._free.append(b)

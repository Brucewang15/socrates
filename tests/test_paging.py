import pytest

from model.paging import BlockAllocator, OutOfBlocks, blocks_for


@pytest.mark.parametrize("tokens,expected", [
    (0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (2047, 128), (2048, 128), (2049, 129),
])
def test_blocks_for(tokens, expected):
    assert blocks_for(tokens, 16) == expected


def test_alloc_free_round_trip():
    a = BlockAllocator(10)
    got = a.alloc(4)
    assert len(got) == len(set(got)) == 4
    assert a.n_free == 6
    a.free(got)
    assert a.n_free == 10
    # everything comes back, and can be taken again in full
    assert sorted(a.alloc(10)) == list(range(1, 11))


def test_null_block_is_never_handed_out():
    a = BlockAllocator(50)
    assert 0 not in a.alloc(50)
    with pytest.raises(ValueError):
        a.free([0])


def test_exhaustion_raises_and_changes_nothing():
    a = BlockAllocator(3)
    a.alloc(2)
    with pytest.raises(OutOfBlocks):
        a.alloc(2)
    assert a.n_free == 1


def test_double_free_is_caught_and_changes_nothing():
    a = BlockAllocator(4)
    got = a.alloc(2)
    a.free(got)
    with pytest.raises(ValueError):
        a.free(got)
    with pytest.raises(ValueError):
        a.free([got[0], got[0]])
    assert a.n_free == 4


def test_freed_blocks_are_reused_first():
    a = BlockAllocator(8)
    first = a.alloc(3)
    a.alloc(2)
    a.free(first)
    assert sorted(a.alloc(3)) == sorted(first)

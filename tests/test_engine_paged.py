"""The paged engine end to end on the tiny model: more requests than rows.

float64 so that batch invariance is a real assertion rather than a coin toss on
argmax near-ties: the same request must produce the same tokens whether it ran
alone or squeezed in beside seventeen others.
"""

import random

import torch

from model.inference_paged import Engine
from model.qwen.qwen_kv_paged import Qwen3Paged
from tests.tiny import StubTokenizer, build, random_state

MAX_LEN, MAX_NEW = 96, 24


def engine(max_batch=18):
    sd = random_state(Qwen3Paged, dtype=torch.float64)
    model = build(Qwen3Paged, sd, dtype=torch.float64)
    return Engine(model=model, tok=StubTokenizer(), device="cpu", compile=False,
                  max_batch=max_batch, max_len=MAX_LEN, max_new_tokens=MAX_NEW,
                  block_size=8)


def prompts(n=24, seed=0):
    rng = random.Random(seed)
    return ["".join(chr(rng.randint(33, 126)) for _ in range(rng.randint(3, 40)))
            for _ in range(n)]


def check_no_block_shared(e):
    owned = [b for r in range(e.n_active) for b in e.cache.row_blocks[r]]
    assert len(owned) == len(set(owned))
    assert 0 not in owned
    assert e.allocator.n_free + len(owned) == e.allocator.num_blocks
    for r in range(e.n_active, e.max_batch):
        assert e.cache.row_blocks[r] == []
        assert e.cache.block_table[r].abs().sum() == 0


def test_more_requests_than_rows_all_complete_and_every_block_returns():
    e = engine()
    reqs = [e.submit(p) for p in prompts()]
    steps = 0
    while e.pending or e.n_active:          # Engine.run, checked every iteration
        e.admit()
        e.sweep()
        check_no_block_shared(e)
        if e.n_active:
            e.decode_step()
            steps += 1
        e.sweep()
        check_no_block_shared(e)
    assert all(r.done for r in reqs)
    assert e.allocator.n_free == e.allocator.num_blocks
    assert steps > 0


def test_batched_output_equals_each_request_run_alone():
    ps = prompts(n=22, seed=3)
    e = engine()
    batched = [e.submit(p) for p in ps]
    e.run()

    for p, b in zip(ps, batched):
        solo = engine(max_batch=1)
        r = solo.submit(p)
        solo.run()
        assert r.output == b.output, p


def test_cancelled_rows_are_retired_and_freed():
    e = engine(max_batch=4)
    reqs = [e.submit(p) for p in prompts(n=6)]
    e.admit()
    reqs[1].cancelled = True
    e.decode_step()
    e.sweep()
    assert reqs[1].row == 1 and reqs[1] not in e.rows
    check_no_block_shared(e)
    e.run()
    assert e.allocator.n_free == e.allocator.num_blocks

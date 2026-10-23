"""Two markers keep the default run laptop-safe.

    uv run pytest                     # everything that runs on a Mac
    uv run pytest -m cuda             # on the GPU host, needs the built extension
    SOCRATES_WEIGHTS=1 uv run pytest -m weights   # loads the 8 GB checkpoint
"""

import os

import pytest
import torch


def _cuda_kernel() -> bool:
    if not torch.cuda.is_available():
        return False
    from model import kernels
    return kernels.cuda_kernel_loaded()


def pytest_collection_modifyitems(config, items):
    cuda_ok = _cuda_kernel()
    weights_ok = os.getenv("SOCRATES_WEIGHTS") == "1"
    no_cuda = pytest.mark.skip(reason="needs CUDA and the built socrates kernel")
    no_weights = pytest.mark.skip(reason="set SOCRATES_WEIGHTS=1 to load the checkpoint")
    for item in items:
        if "cuda" in item.keywords and not cuda_ok:
            item.add_marker(no_cuda)
        if "weights" in item.keywords and not weights_ok:
            item.add_marker(no_weights)

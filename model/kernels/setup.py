"""Build the socrates CUDA kernels next to this file.

    cd model/kernels && python setup.py build_ext --inplace    # -> _C.abi3.so

Needs an nvcc whose major version matches torch.version.cuda (13 for the locked
torch 2.14.0), which is why it runs in the Dockerfile's `kernels` stage and not
on a laptop. Compiled for the L4 only (sm_89); set TORCH_CUDA_ARCH_LIST to build
for something else.

The .so has no Python bindings -- see csrc/ops.cpp -- so it is built against the
limited API and loaded with torch.ops.load_library by model/kernels/__init__.py.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

setup(
    name="socrates_kernels",
    ext_modules=[
        CUDAExtension(
            "_C",
            ["csrc/ops.cpp", "csrc/paged_attention.cu"],
            extra_compile_args={
                # py_limited_api=True already defines Py_LIMITED_API for both
                # compilers; defining it again only earns a redefinition warning.
                "cxx": ["-O3"],
                # -lineinfo maps profiler samples back to source lines, at no
                # cost to the generated code.
                "nvcc": ["-O3", "-lineinfo"],
            },
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

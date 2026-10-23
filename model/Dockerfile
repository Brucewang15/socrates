# GPU serving. Built from the repo root, not from model/:
#
#   docker build -f model/Dockerfile -t socrates-model .
#   docker run --gpus all -p 8080:8080 -v hf:/weights socrates-model
#
# python:3.12-slim rather than a CUDA base image on purpose: the torch wheel
# ships its own CUDA runtime libraries, and nvidia-container-toolkit injects
# the host driver. A CUDA base would add ~2 GB of libraries torch ignores.
#
# Three stages since day 8, and the runtime image is still exactly that:
#
#   base     python, a C compiler, and the locked dependencies
#   kernels  base + just enough of the CUDA toolkit to run nvcc; builds
#            model/kernels/_C.abi3.so (paged attention) for the L4
#   runtime  base + the source + that one .so
#
# The kernel stage is built FROM base, not from a CUDA image, so it compiles
# against the very torch, glibc and Python the runtime loads it into. The
# toolkit is copied in from NVIDIA's image instead -- the major version has to
# match torch.version.cuda (13 for torch 2.14.0) or cpp_extension refuses.
FROM python:3.12-slim AS base

# torch.compile needs a C compiler at *run* time, not build time: Inductor emits
# C++/Triton source on the first decode step and shells out to cc to build it.
# Without this the compile raises InductorError, the serve thread dies, and
# /api/chat hangs forever while /health still reports ok -- see model/server.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies in their own layer so editing the engine does not re-resolve
# and re-download torch. dev is pytest, so the CUDA tests run in this image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra model --extra dev


# Only the pieces of the toolkit nvcc needs to compile one extension: the driver
# binaries, the NVVM front end, headers, and cudart to link against. The full
# devel tree is several GB, almost all of it static math libraries this build
# never touches, and the GPU host's disk has filled before (docs/day_7_docs.md).
# The layout under targets/ is kept as-is because nvcc.profile hardcodes it.
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04 AS cuda
RUN set -eux; \
    arch="$(ls /usr/local/cuda/targets)"; src=/usr/local/cuda; dst=/opt/cuda; \
    mkdir -p "$dst/targets/$arch/lib"; \
    cp -a "$src/bin" "$src/nvvm" "$dst/"; \
    cp -a "$src/targets/$arch/include" "$dst/targets/$arch/"; \
    cp -a "$src/targets/$arch/lib/"libcudart.so* "$dst/targets/$arch/lib/"; \
    ln -s "targets/$arch/include" "$dst/include"; \
    ln -s "targets/$arch/lib" "$dst/lib64"; \
    du -sh "$dst"


FROM base AS kernels
COPY --from=cuda /opt/cuda /usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:$PATH" \
    TORCH_CUDA_ARCH_LIST=8.9
# Only the kernel sources, so editing the engine does not recompile the kernel.
COPY model/kernels/setup.py model/kernels/
COPY model/kernels/csrc/ model/kernels/csrc/
RUN cd model/kernels \
    && nvcc --version | tail -2 \
    && python setup.py build_ext --inplace \
    && ls -la _C*.so


FROM base AS runtime
COPY model/ model/
COPY --from=kernels /app/model/kernels/_C*.so model/kernels/

# The benches run the same engine against the same weights, so they ride in
# this image rather than getting one of their own -- numpy is already here via
# torch, and plots.py (the only matplotlib consumer) runs on your laptop from
# the .npz. bench/results is dockerignored: results are produced, not shipped.
COPY bench/ bench/
COPY analysis/ analysis/
COPY tests/ tests/

ENV PYTHONPATH=/app \
    DEVICE=cuda \
    HF_HOME=/weights

# Loading the .so needs no GPU -- it only runs the op registrations -- so a
# kernel that cannot load (wrong torch, wrong glibc, missing symbol) fails the
# build here instead of failing the first decode step in production.
RUN python -c "import glob, torch; \
so = glob.glob('/app/model/kernels/_C*.so'); assert len(so) == 1, so; \
torch.ops.load_library(so[0]); \
print('kernel ok:', torch.ops.socrates.paged_attention_decode.default._schema)"

# Weights are NOT baked into the image!
VOLUME /weights
EXPOSE 8080

# One worker, always. Each extra worker is another full copy of the model on
# the same card.
CMD ["uvicorn", "model.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# GPU serving. Built from the repo root, not from model/:
#
#   docker build -f model/Dockerfile -t socrates-model .
#   docker run --gpus all -p 8080:8080 -v hf:/weights socrates-model
#
# python:3.12-slim rather than a CUDA base image on purpose: the torch wheel
# ships its own CUDA runtime libraries, and nvidia-container-toolkit injects
# the host driver. A CUDA base would add ~2 GB of libraries torch ignores.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies in their own layer so editing the engine does not re-resolve
# and re-download torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra model

COPY model/ model/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    DEVICE=cuda \
    HF_HOME=/weights

# Weights are NOT baked into the image!
VOLUME /weights
EXPOSE 8080

# One worker, always. Each extra worker is another full copy of the model on
# the same card.
CMD ["uvicorn", "model.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# Backend. Built from the repo root, not from backend/:
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY backend/ backend/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    MODEL_URL=http://model:8080

EXPOSE 8000

# Raise --workers with the instance's core count.
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

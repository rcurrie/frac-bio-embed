# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed for hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Use uv to install dependencies into a specific path
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

# Install into /app/.venv to make it easy to copy
# --no-editable ensures we don't have symlinks that break across stages
RUN uv sync --frozen --no-dev --no-install-project --no-editable

# --- Stage 2: Final Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Only copy the installed packages from the builder
# This leaves behind g++, the uv cache, and apt metadata
COPY --from=builder /app/.venv /app/.venv

# Ensure the virtualenv is on the PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV S3_BASE=""

# Copy application code and the specific fractal source files
COPY train.py ./
# Copy fractal-embeddings files directly into the working directory
COPY fractal-embeddings/moonshot-fractal-embeddings/src/fractal_v5.py \
     fractal-embeddings/moonshot-fractal-embeddings/src/multi_model_pipeline.py \
     ./

# Entrypoint doesn't need 'uv run' anymore since we are using the venv's python
ENTRYPOINT ["python", "train.py"]
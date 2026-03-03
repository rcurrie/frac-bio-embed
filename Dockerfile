FROM python:3.12-slim

WORKDIR /app

# Build tools for hnswlib C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no venv needed in container)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY train.py ./

# Copy fractal-embeddings submodule (only the src we need)
COPY fractal-embeddings/moonshot-fractal-embeddings/src/fractal_v5.py \
     fractal-embeddings/moonshot-fractal-embeddings/src/multi_model_pipeline.py \
     fractal-embeddings/moonshot-fractal-embeddings/src/

# S3_BASE can be set via env var or overridden on command line
ENV S3_BASE=""

ENTRYPOINT ["uv", "run", "python", "train.py"]

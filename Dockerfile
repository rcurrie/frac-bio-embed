# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

# Train-only deps (skip ingest group)
RUN uv sync --frozen --no-dev --no-group ingest --no-install-project --no-editable

# --- Stage 2: Final Runtime ---
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /app/.venv/bin /app/.venv/bin

# Chunked site-packages: split large packages into separate layers
# so no single layer is too big to push reliably.
WORKDIR /app/.venv/lib/python3.12/site-packages/

# NVIDIA/CUDA (~200-500MB each)
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cublas*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cuda_cupti*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cuda_nvrtc*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cuda_runtime*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cudnn*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cufft*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_curand*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cusolver*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cusparse*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_cusparselt*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_nccl*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_nvjitlink*/ ./
COPY --from=builder /app/.venv/lib/python3.12/site-packages/nvidia_nvtx*/ ./

# Triton
COPY --from=builder /app/.venv/lib/python3.12/site-packages/triton/ ./triton/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/triton-*.dist-info/ ./

# PyTorch
COPY --from=builder /app/.venv/lib/python3.12/site-packages/torch/ ./torch/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/torch-*.dist-info/ ./

# NumPy
COPY --from=builder /app/.venv/lib/python3.12/site-packages/numpy/ ./numpy/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/numpy-*.dist-info/ ./

# PyArrow
COPY --from=builder /app/.venv/lib/python3.12/site-packages/pyarrow/ ./pyarrow/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/pyarrow-*.dist-info/ ./

# boto3 / botocore / s3fs
COPY --from=builder /app/.venv/lib/python3.12/site-packages/boto3/ ./boto3/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/botocore/ ./botocore/
COPY --from=builder /app/.venv/lib/python3.12/site-packages/s3fs/ ./s3fs/

# Everything else
COPY --from=builder /app/.venv/lib/python3.12/site-packages/ ./

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV S3_BASE=""

COPY train.py ./
COPY fractal-embeddings/moonshot-fractal-embeddings/src/fractal_v5.py \
     fractal-embeddings/moonshot-fractal-embeddings/src/multi_model_pipeline.py \
     ./

ENTRYPOINT ["python", "train.py"]

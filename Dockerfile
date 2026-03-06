# PyTorch official runtime image: Python 3.12, PyTorch, CUDA 12.4, cuDNN 9
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install remaining deps into system Python (torch/numpy already in base)
RUN pip install --no-cache-dir \
    "pyarrow>=15.0.0" \
    "tqdm>=4.66.0" \
    "s3fs>=2026.2.0" \
    "boto3>=1.35.0"

ENV PYTHONUNBUFFERED=1
ENV S3_BASE=""

COPY train.py ./
COPY fractal-embeddings/moonshot-fractal-embeddings/src/fractal_v5.py \
     fractal-embeddings/moonshot-fractal-embeddings/src/multi_model_pipeline.py \
     ./

ENTRYPOINT ["python", "train.py"]

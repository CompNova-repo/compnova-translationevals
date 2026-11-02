

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    USE_TF=NO \
    USE_JAX=NO

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Step 1: Cython pin - fixes an old PyYAML (5.3.1) build failure against
# newer Cython that otherwise breaks unbabel-comet's install.
RUN pip install --no-cache-dir "cython<3.0.0"

# Step 2: PyTorch with CUDA 12.1 wheels (match this to your deployment
# host's driver/CUDA version if different).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Step 3: COMET, pinned to an exact version and isolated with --no-deps,
# to avoid pip's resolver backtracking through years of old comet
# releases trying to satisfy version constraints against newer packages.
RUN pip install --no-cache-dir "unbabel-comet==2.2.7" --no-deps
RUN pip install --no-cache-dir entmax "jsonargparse==3.13.1" pytorch-lightning sacrebleu torchmetrics

# Step 4: force the exact versions COMET needs. This must be the LAST
# thing that touches these six packages before the app runs.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" "protobuf==4.25.9" "huggingface_hub==0.36.2" \
    "tokenizers==0.15.2" "transformers==4.39.3" "torchmetrics==0.10.3" \
    "accelerate==0.27.2" \
    --force-reinstall --no-deps

# Step 5: app requirements, still --no-deps throughout - sentence-
# transformers in particular will silently upgrade transformers /
# huggingface_hub again otherwise, undoing step 4.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt --no-deps
RUN pip install --no-cache-dir av ctranslate2 onnxruntime sentence-transformers safetensors --no-deps

# Step 6: remove optional backends that transformers/sentence-transformers
# try to import opportunistically and crash on with this transformers
# pin - none of them are actually needed (this app only uses PyTorch).
RUN pip uninstall -y jax jaxlib peft tensorflow tensorflow-cpu keras wandb 2>/dev/null || true

# MetricX is research code with no setup.py, not on PyPI - clone it at
# build time and pin to a known-good commit.
RUN git clone https://github.com/google-research/metricx.git /app/metricx \
    && git -C /app/metricx checkout fc4978eb064670f7cc33e93ea4f52d38396b8ae6

COPY eval_core.py api.py ./

EXPOSE 8000

# Pass your Hugging Face token at run time, e.g.:
#   docker run --gpus all -p 8000:8000 -e HF_TOKEN=hf_your_token compnova-eval-api
# (also requires the nvidia-container-toolkit installed on the host for --gpus to work)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    USE_TF=NO \
    USE_JAX=NO \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/var/lib/compnova/hf-cache

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
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Step 3: COMET, pinned to an exact version and isolated with --no-deps,
# to avoid pip's resolver backtracking through years of old comet
# releases trying to satisfy version constraints against newer packages.
RUN pip install --no-cache-dir "unbabel-comet==2.2.7" --no-deps
RUN pip install --no-cache-dir entmax "jsonargparse==3.13.1" pytorch-lightning sacrebleu torchmetrics

# Step 4: app requirements, still --no-deps throughout - sentence-
# transformers in particular will silently upgrade transformers /
# huggingface_hub again otherwise, undoing the pins applied in step 7.
COPY requirements-api.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -c constraints.txt --no-deps
RUN pip install --no-cache-dir av ctranslate2 onnxruntime sentence-transformers safetensors --no-deps

# Step 5: FastAPI/Starlette/Uvicorn/Pydantic's own required sub-dependencies,
# skipped by --no-deps above along with everything else. Web framework + basic
# HTTP internals, not ML packages, so no version conflict risk with the COMET
# pin. Deliberately NOT --no-deps here - let pip pull each package's own small
# dependency tree normally instead of finding gaps one at a time.
RUN pip install --no-cache-dir click h11 starlette anyio sniffio idna \
    pydantic pydantic-core annotated_doc annotated-types requests \
    pytz python-dateutil

# Step 6: `datasets` (imported by metricx24.predict) and `gTTS` (used by
# POST /translate). datasets is installed with --no-deps so it cannot drag
# huggingface_hub forward; its own runtime dependencies are listed explicitly.
RUN pip install --no-cache-dir datasets --no-deps \
    && pip install --no-cache-dir pyarrow dill xxhash multiprocess fsspec tqdm pandas \
    && pip install --no-cache-dir gTTS beautifulsoup4

# Step 7: force the exact versions COMET needs. This must be AFTER every
# package install that might have touched them.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" "protobuf==4.25.9" "huggingface_hub==0.36.2" \
    "tokenizers==0.15.2" "transformers==4.39.3" "torchmetrics==0.10.3" \
    "accelerate==0.27.2" \
    --force-reinstall --no-deps

# Step 8: remove optional backends that transformers/sentence-transformers
# try to import opportunistically and crash on with this transformers
# pin - none of them are actually needed (this app only uses PyTorch).
RUN pip uninstall -y jax jaxlib peft tensorflow tensorflow-cpu keras wandb 2>/dev/null || true

# MetricX is research code with no setup.py, not on PyPI - clone it at
# build time and pin to a known-good commit.
RUN git clone https://github.com/google-research/metricx.git /app/metricx \
    && git -C /app/metricx checkout fc4978eb064670f7cc33e93ea4f52d38396b8ae6

# The application. eval_store.py, config.py and errors.py MUST be copied -
# api.py imports all three at module level and the container dies at
# startup without them.
COPY eval_core.py eval_store.py config.py errors.py api.py ./
COPY cron/ ./cron/

# Where the SQLite store and the upload spool live. Mount a volume here, or
# every pod restart loses every evaluation.
RUN mkdir -p /var/lib/compnova/hf-cache /var/lib/compnova/spool
VOLUME ["/var/lib/compnova"]

ENV EVAL_DB_PATH=/var/lib/compnova/evaluations.db \
    EVAL_SPOOL_DIR=/var/lib/compnova/spool \
    METRICX_DIR=/app/metricx

EXPOSE 8000

# Liveness only - the models take ~10 minutes to load on a cold start and
# /health/ready would report the container unhealthy the whole time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Pass your Hugging Face token at run time, e.g.:
#   docker run --gpus all -p 8000:8000 -e HF_TOKEN=hf_your_token compnova-eval-api
# (also requires the nvidia-container-toolkit installed on the host for --gpus to work)
#
# Set EVAL_API_KEYS to require an X-API-Key header:
#   docker run ... -e EVAL_API_KEYS=some-long-random-string compnova-eval-api
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

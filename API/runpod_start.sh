#!/usr/bin/env bash
# Bring the S2ST evaluation API and dashboard up on a RunPod pod (or any Linux GPU box).
#
# Usage (from anywhere):
#   bash /workspace/compnova-translationevals/API/runpod_start.sh            # set up and start
#   bash /workspace/compnova-translationevals/API/runpod_start.sh --restart  # restart both services (after a git pull)
#   bash /workspace/compnova-translationevals/API/runpod_start.sh --check    # report status only, change nothing
#
# Safe to re-run at any time. Every step checks first and only acts if needed:
# on a brand-new pod it installs and downloads everything; on a running pod it
# just confirms things are healthy. See RUNPOD_SETUP.md for why each step exists.
#
# Optional settings (environment variables):
#   HF_TOKEN            Hugging Face token. Needed once per pod to download the gated
#                       COMET-Kiwi model. If unset, the script looks for a saved token,
#                       then asks for one.
#   S2ST_MODEL_CACHE    Where models are downloaded (default /root/hf-cache, on the pod's
#                       own disk; the shared /workspace volume proved unreliable for this).
#   S2ST_LOG_DIR        Where logs go (default /root/logs).
#   EVAL_DB_PATH        Where results are stored (default API/evaluations.db).
set -euo pipefail

MODE=start
case "${1:-}" in
  --restart) MODE=restart ;;
  --check)   MODE=check ;;
  "")        ;;
  *) echo "Unknown option: $1 (use --restart or --check)"; exit 2 ;;
esac

# ---------- settings ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
PY="${PYTHON:-python3}"
MODEL_CACHE="${S2ST_MODEL_CACHE:-/root/hf-cache}"
SHARED_TOKEN_FILE=/workspace/models/huggingface/token
METRICX_DIR=/opt/metricx
METRICX_REPO=https://github.com/google-research/metricx.git
METRICX_COMMIT=fc4978eb064670f7cc33e93ea4f52d38396b8ae6
API_PORT=8000
UI_PORT=8501

# Logs go on the pod's own disk: the shared /workspace volume shows new writes late.
LOG_DIR="${S2ST_LOG_DIR:-/root/logs}"
mkdir -p "$LOG_DIR"
API_LOG="$LOG_DIR/api.log"
UI_LOG="$LOG_DIR/ui.log"
CONSTRAINTS="$LOG_DIR/constraints.txt"

say()  { echo "== $*"; }
info() { echo "   $*"; }
fail() { echo; echo "FAILED: $*"; exit 1; }

# Only one copy of this script at a time.
exec 9>/tmp/s2st_start.lock
flock -n 9 || fail "runpod_start.sh is already running in another terminal."

# ---------- environment ----------
# The pod sets GIT_CONFIG_PARAMETERS in a format git can't parse.
unset GIT_CONFIG_PARAMETERS || true
export HF_HOME="$MODEL_CACHE"
export USE_TF=0 USE_TORCH=1
# RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer, which makes every download fail.
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_CACHE_DIR=/root/.cache/pip
# Pulkit's patch makes Python ignore chmod errors on /workspace. Use it if it's there.
if [ -f /workspace/python_patches/sitecustomize.py ]; then
  export PYTHONPATH=/workspace/python_patches
else
  unset PYTHONPATH || true
fi

cd "$REPO_DIR"
[ -f api.py ] && [ -f eval_core.py ] || fail "api.py / eval_core.py not found in $REPO_DIR. Run this script from the repo's API folder."

api_up() { curl -sf "localhost:$API_PORT/health" >/dev/null 2>&1; }
ui_up()  { curl -sf -o /dev/null "localhost:$UI_PORT" 2>/dev/null; }

if [ "$MODE" = check ]; then
  say "Status"
  if api_up; then info "API: running"; else info "API: NOT running"; fi
  if ui_up;  then info "Dashboard: running"; else info "Dashboard: NOT running"; fi
  info "Logs: $API_LOG, $UI_LOG"
  exit 0
fi

# ---------- 1. system checks ----------
say "1/8 System checks"
for tool in git curl; do
  if ! command -v "$tool" >/dev/null; then
    info "installing $tool"
    apt-get update -qq && apt-get install -y -qq "$tool" >/dev/null
  fi
done
PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
  3.10|3.11|3.12) info "Python $PYVER" ;;
  *) info "WARNING: Python $PYVER. This stack is tested on 3.10 to 3.12; installs may fail." ;;
esac
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  info "WARNING: no NVIDIA GPU visible. Everything will run on CPU and be very slow."
fi
FREE_GB=$(df -BG --output=avail "$(dirname "$MODEL_CACHE")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 15 ]; then
  fail "only ${FREE_GB} GB free for the model cache at $MODEL_CACHE; about 15 GB is needed. Set S2ST_MODEL_CACHE to a bigger disk."
fi

# ---------- 2. shell defaults for new terminals ----------
say "2/8 Shell defaults for new terminals"
BASHRC_BLOCK="# S2ST env (added by runpod_start.sh)
unset GIT_CONFIG_PARAMETERS
export HF_HOME=$MODEL_CACHE
export USE_TF=0 USE_TORCH=1
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -f /workspace/python_patches/sitecustomize.py ] && export PYTHONPATH=/workspace/python_patches
# end S2ST env"
touch ~/.bashrc
if grep -q "^# end S2ST env" ~/.bashrc; then
  sed -i '/^# S2ST env (added by runpod_start.sh)/,/^# end S2ST env/d' ~/.bashrc
elif grep -q "^# S2ST env (added by runpod_start.sh)" ~/.bashrc; then
  # Block written by the first version of this script (no end marker).
  sed -i '/^# S2ST env (added by runpod_start.sh)/,/^export USE_TF=0 USE_TORCH=1/d' ~/.bashrc
fi
printf '%s\n' "$BASHRC_BLOCK" >> ~/.bashrc
info "~/.bashrc updated"

# ---------- 3. Python packages ----------
say "3/8 Python packages"
cat > "$CONSTRAINTS" <<'EOF'
numpy==1.26.4
transformers==4.39.3
tokenizers==0.15.2
protobuf==4.25.9
accelerate==0.27.2
huggingface_hub==0.36.2
torchmetrics==0.10.3
torch==2.8.0
EOF

# Packages that crash on import under these pins (transformers imports them automatically).
for pkg in jax jaxlib flax optax orbax-checkpoint jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt \
           tensorflow tensorflow-probability tensorflow-metadata tensorflow-hub tensorflow-datasets tensorflow-text \
           tensorboard tf-keras keras ydf wandb peft; do
  if "$PY" -m pip show "$pkg" >/dev/null 2>&1; then
    info "removing $pkg"
    "$PY" -m pip uninstall -y -q "$pkg"
  fi
done

TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo none)
case "$TORCH_VER" in
  2.8.0*) info "torch $TORCH_VER" ;;
  *) info "installing torch 2.8.0 (found: $TORCH_VER)"
     "$PY" -m pip install -q "torch==2.8.0" ;;
esac

if ! "$PY" -c "import comet, faster_whisper, sentence_transformers, fastapi, uvicorn, streamlit, altair, datasets, librosa, sacrebleu, entmax" >/dev/null 2>&1; then
  info "installing the evaluation stack (a few minutes on a new pod)"
  "$PY" -m pip install -q "cython<3.0.0" -c "$CONSTRAINTS"
  grep -viE "^\s*(sentence-transformers|unbabel-comet)" requirements-api.txt > /tmp/s2st-req-api.txt
  "$PY" -m pip install -q -r /tmp/s2st-req-api.txt -c "$CONSTRAINTS"
  "$PY" -m pip install -q entmax "jsonargparse==3.13.1" pytorch-lightning sacrebleu scikit-learn \
      datasets streamlit altair requests pandas -c "$CONSTRAINTS"
  # Their declared requirements conflict with the pins, but these exact versions work with them.
  "$PY" -m pip install -q --no-deps "unbabel-comet==2.2.7" "sentence-transformers==5.7.0"
else
  info "evaluation stack present"
fi

WRONG=$("$PY" - <<'EOF'
from importlib import metadata
want = {"numpy": "1.26.4", "transformers": "4.39.3", "tokenizers": "0.15.2",
        "protobuf": "4.25.9", "accelerate": "0.27.2", "huggingface_hub": "0.36.2",
        "torchmetrics": "0.10.3", "unbabel-comet": "2.2.7", "sentence-transformers": "5.7.0"}
bad = []
for name, version in want.items():
    try:
        if metadata.version(name) != version:
            bad.append(f"{name}=={version}")
    except metadata.PackageNotFoundError:
        bad.append(f"{name}=={version}")
print(" ".join(bad))
EOF
)
if [ -n "$WRONG" ]; then
  info "re-pinning: $WRONG"
  "$PY" -m pip install -q --no-deps $WRONG
fi

"$PY" -c "import comet, faster_whisper, sentence_transformers, fastapi, streamlit, datasets" >/dev/null 2>&1 \
  || { "$PY" -c "import comet, faster_whisper, sentence_transformers, fastapi, streamlit, datasets"; fail "packages still don't import (error above)."; }
info "all packages import"

# ---------- 4. library fixes ----------
say "4/8 Library fixes"
# PyAV 14+ removed an argument that faster-whisper 1.2.1 still passes.
FW_AUDIO=$("$PY" -c "import faster_whisper, os; print(os.path.join(os.path.dirname(faster_whisper.__file__), 'audio.py'))")
if grep -q "metadata_errors" "$FW_AUDIO"; then
  sed -i 's/, metadata_errors="ignore"//' "$FW_AUDIO"
  info "patched faster-whisper for PyAV"
else
  info "faster-whisper already patched"
fi
# CTranslate2 (Whisper) needs the CUDA 12 libraries that torch installs via pip.
CUDA_LIBS=$("$PY" - <<'EOF'
import importlib.util, os
dirs = []
for pkg in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        spec = importlib.util.find_spec(pkg)
    except ModuleNotFoundError:
        spec = None
    if spec and spec.submodule_search_locations:
        lib = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        if os.path.isdir(lib):
            dirs.append(lib)
print(":".join(dirs))
EOF
)
if [ -n "$CUDA_LIBS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  info "CUDA 12 libraries on LD_LIBRARY_PATH"
fi

# ---------- 5. MetricX ----------
say "5/8 MetricX at the pinned commit"
if [ ! -f "$METRICX_DIR/metricx24/predict.py" ]; then
  info "cloning into $METRICX_DIR (container disk; git can't write to /workspace)"
  rm -rf "$METRICX_DIR"
  git clone -q "$METRICX_REPO" "$METRICX_DIR"
fi
if [ "$(git -C "$METRICX_DIR" rev-parse HEAD)" != "$METRICX_COMMIT" ]; then
  git -C "$METRICX_DIR" -c advice.detachedHead=false checkout -q "$METRICX_COMMIT"
fi
info "at $(git -C "$METRICX_DIR" rev-parse --short HEAD)"
# API/metricx must point at it. Replace a broken link or an empty leftover folder.
if [ -L metricx ] && [ "$(readlink metricx)" != "$METRICX_DIR" ]; then rm -f metricx; fi
if [ -d metricx ] && [ ! -L metricx ] && [ ! -f metricx/metricx24/predict.py ]; then rm -rf metricx; fi
if [ ! -e metricx ]; then
  ln -s "$METRICX_DIR" metricx
  info "linked API/metricx -> $METRICX_DIR"
fi
[ -f metricx/metricx24/predict.py ] || fail "API/metricx does not lead to MetricX. Check: ls -la $REPO_DIR/metricx"

# ---------- 6. models ----------
say "6/8 Models in $MODEL_CACHE"
mkdir -p "$MODEL_CACHE"
if [ -z "${HF_TOKEN:-}" ]; then
  for f in "$MODEL_CACHE/token" "$SHARED_TOKEN_FILE"; do
    if [ -s "$f" ]; then HF_TOKEN=$(tr -d '[:space:]' < "$f"); info "using saved token from $f"; break; fi
  done
fi
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -t 0 ]; then
    read -rsp "   Hugging Face token (input hidden): " HF_TOKEN; echo
  else
    fail "no Hugging Face token. Run with HF_TOKEN=... set."
  fi
fi
export HF_TOKEN
( umask 077; printf '%s' "$HF_TOKEN" > "$MODEL_CACHE/token" )

HF_HUB_DISABLE_XET=1 "$PY" - <<'EOF' || fail "model download failed (see above)."
import os, sys
from huggingface_hub import snapshot_download

MODELS = [
    ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),
    ("Unbabel/wmt22-cometkiwi-da", None),
    ("microsoft/infoxlm-large", None),
    ("google/metricx-24-hybrid-large-v2p6-bfloat16", None),
    ("google/mt5-xl", ["*.json", "*.model"]),   # MetricX only needs the tokenizer
    ("sentence-transformers/LaBSE", None),
    ("tabularisai/multilingual-sentiment-analysis", None),
]

# Weight formats this project never loads.
SKIP = ["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot", "rust_model*", "openvino/*", "*.tflite"]

def empty_files(path):
    return [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
            if os.path.getsize(os.path.join(r, f)) == 0]

failed = []
for repo, patterns in MODELS:
    ok = False
    for attempt in (1, 2):
        try:
            path = snapshot_download(repo, allow_patterns=patterns, ignore_patterns=SKIP,
                                     force_download=(attempt == 2))
        except Exception as e:
            msg = str(e).splitlines()[0][:200]
            if "gated" in type(e).__name__.lower() or "401" in msg or "403" in msg:
                print(f"   {repo}: access denied. Accept its terms on huggingface.co with the account that owns this token.")
                break
            print(f"   {repo}: download error ({type(e).__name__}: {msg}); retrying" if attempt == 1 else f"   {repo}: download error again")
            continue
        bad = empty_files(path)
        if not bad:
            print(f"   {repo}: ok")
            ok = True
            break
        print(f"   {repo}: {len(bad)} empty file(s), re-downloading")
    if not ok:
        failed.append(repo)

if failed:
    print("   could not get: " + ", ".join(failed))
    sys.exit(1)
EOF

# ---------- 7. API ----------
say "7/8 API on port $API_PORT"
if [ "$MODE" = restart ]; then
  pkill -f "[u]vicorn api:app" || true
  pkill -f "[s]treamlit run" || true
  sleep 3
fi
if api_up; then
  info "already running"
else
  pkill -f "[u]vicorn api:app" || true
  sleep 2
  setsid nohup "$PY" -m uvicorn api:app --host 0.0.0.0 --port "$API_PORT" > "$API_LOG" 2>&1 < /dev/null 9>&- &
  echo -n "   loading models"
  sleep 5
  for _ in $(seq 1 180); do
    if api_up; then break; fi
    if ! pgrep -f "[u]vicorn api:app" >/dev/null && { sleep 3; ! pgrep -f "[u]vicorn api:app" >/dev/null; }; then
      echo; echo "   The API process exited. Last lines of $API_LOG:"; tail -30 "$API_LOG"
      fail "API did not start."
    fi
    echo -n "."
    sleep 5
  done
  echo
  api_up || { tail -30 "$API_LOG"; fail "API still not answering after 15 minutes."; }
  info "started"
fi

# ---------- 8. dashboard ----------
say "8/8 Dashboard on port $UI_PORT"
if ui_up; then
  info "already running"
else
  pkill -f "[s]treamlit run" || true
  setsid nohup "$PY" -m streamlit run DemoApp.py --server.port "$UI_PORT" --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false > "$UI_LOG" 2>&1 < /dev/null 9>&- &
  sleep 3
  for _ in $(seq 1 30); do
    if ui_up; then break; fi
    if ! pgrep -f "[s]treamlit run" >/dev/null && { sleep 3; ! pgrep -f "[s]treamlit run" >/dev/null; }; then break; fi
    sleep 2
  done
  ui_up || { echo "   Last lines of $UI_LOG:"; tail -20 "$UI_LOG"; fail "dashboard did not start."; }
  info "started"
fi

echo
echo "All up."
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "  Dashboard: https://${RUNPOD_POD_ID}-${UI_PORT}.proxy.runpod.net"
  echo "  API docs:  https://${RUNPOD_POD_ID}-${API_PORT}.proxy.runpod.net/docs"
  echo "  (If a URL says 'Waiting for service', add port $API_PORT / $UI_PORT as an HTTP port in the pod settings.)"
else
  echo "  Dashboard: http://localhost:${UI_PORT}"
  echo "  API docs:  http://localhost:${API_PORT}/docs"
fi
echo "  Logs:      $API_LOG and $UI_LOG"

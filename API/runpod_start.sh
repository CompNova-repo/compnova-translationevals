#!/usr/bin/env bash
# Bring the S2ST evaluation API and dashboard up on the RunPod pod.
#
# Usage (from anywhere on the pod):
#   bash /workspace/compnova-translationevals/API/runpod_start.sh
#
# Safe to re-run at any time. Every step checks first and only acts if needed,
# so after a pod restart it rebuilds what was lost, and on a healthy pod it
# just confirms everything is running. See RUNPOD_SETUP.md for the reasons
# behind each step.
set -euo pipefail

REPO_DIR=/workspace/compnova-translationevals/API
METRICX_DIR=/opt/metricx
METRICX_REPO=https://github.com/google-research/metricx.git
METRICX_COMMIT=fc4978eb064670f7cc33e93ea4f52d38396b8ae6
CONSTRAINTS=/workspace/constraints.txt
API_LOG=/workspace/api.log
UI_LOG=/workspace/ui.log

# Environment. The pod's own PYTHONPATH setting is broken (it holds a literal
# shell expression), so it is overridden here.
unset GIT_CONFIG_PARAMETERS || true
export HF_HOME=/workspace/models/huggingface
export PYTHONPATH=/workspace/python_patches
export USE_TF=0 USE_TORCH=1

cd "$REPO_DIR"

echo "== 1/6 Shell defaults for new terminals"
if ! grep -q "S2ST env" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'
# S2ST env (added by runpod_start.sh)
unset GIT_CONFIG_PARAMETERS
export HF_HOME=/workspace/models/huggingface
export PYTHONPATH=/workspace/python_patches
export USE_TF=0 USE_TORCH=1
EOF
  echo "   added to ~/.bashrc"
else
  echo "   already present"
fi

echo "== 2/6 Python package pins"
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

# Remove packages that break imports under these pins, if anything brought them back.
for pkg in jax jaxlib flax tensorflow tensorflow-probability tf-keras keras wandb peft; do
  if pip show "$pkg" >/dev/null 2>&1; then
    echo "   removing $pkg"
    pip uninstall -y -q "$pkg"
  fi
done

WRONG=$(python - <<'EOF'
from importlib import metadata
want = {"numpy": "1.26.4", "transformers": "4.39.3", "tokenizers": "0.15.2",
        "protobuf": "4.25.9", "accelerate": "0.27.2", "huggingface_hub": "0.36.2",
        "torchmetrics": "0.10.3"}
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
  echo "   re-pinning: $WRONG"
  pip install -q --no-deps $WRONG
else
  echo "   all pins correct"
fi

if ! python -c "import datasets" >/dev/null 2>&1; then
  echo "   installing datasets (needed by MetricX)"
  pip install -q datasets -c "$CONSTRAINTS"
fi

echo "== 3/6 faster-whisper patch for PyAV 14+"
FW_AUDIO=$(python -c "import faster_whisper, os; print(os.path.join(os.path.dirname(faster_whisper.__file__), 'audio.py'))")
if grep -q "metadata_errors" "$FW_AUDIO"; then
  sed -i 's/, metadata_errors="ignore"//' "$FW_AUDIO"
  echo "   patched $FW_AUDIO"
else
  echo "   already patched"
fi

echo "== 4/6 MetricX at the pinned commit"
if [ ! -f "$METRICX_DIR/metricx24/predict.py" ]; then
  echo "   cloning into $METRICX_DIR (container disk; /workspace does not allow git)"
  rm -rf "$METRICX_DIR"
  git clone -q "$METRICX_REPO" "$METRICX_DIR"
fi
if [ "$(git -C "$METRICX_DIR" rev-parse HEAD)" != "$METRICX_COMMIT" ]; then
  git -C "$METRICX_DIR" -c advice.detachedHead=false checkout -q "$METRICX_COMMIT"
fi
echo "   at $(git -C "$METRICX_DIR" rev-parse --short HEAD)"
if [ ! -e metricx ]; then
  ln -s "$METRICX_DIR" metricx
  echo "   linked API/metricx -> $METRICX_DIR"
fi

echo "== 5/6 API on port 8000"
if curl -sf localhost:8000/health >/dev/null; then
  echo "   already running"
else
  pkill -f "[u]vicorn api:app" || true
  sleep 2
  setsid nohup uvicorn api:app --host 0.0.0.0 --port 8000 > "$API_LOG" 2>&1 < /dev/null &
  echo -n "   loading models"
  for _ in $(seq 1 120); do
    if curl -sf localhost:8000/health >/dev/null; then break; fi
    echo -n "."
    sleep 5
  done
  echo
  if ! curl -sf localhost:8000/health >/dev/null; then
    echo "   API did not start. Last lines of $API_LOG:"
    tail -30 "$API_LOG"
    exit 1
  fi
  echo "   started"
fi

echo "== 6/6 Dashboard on port 8501"
if curl -sf -o /dev/null localhost:8501; then
  echo "   already running"
else
  pkill -f "[s]treamlit run" || true
  setsid nohup streamlit run DemoApp.py --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true --server.enableCORS false --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false > "$UI_LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 12); do
    if curl -sf -o /dev/null localhost:8501; then break; fi
    sleep 2
  done
  if ! curl -sf -o /dev/null localhost:8501; then
    echo "   dashboard did not start. Last lines of $UI_LOG:"
    tail -20 "$UI_LOG"
    exit 1
  fi
  echo "   started"
fi

POD="${RUNPOD_POD_ID:-<pod-id>}"
echo
echo "All up."
echo "  Dashboard: https://${POD}-8501.proxy.runpod.net"
echo "  API docs:  https://${POD}-8000.proxy.runpod.net/docs"
echo "  Logs:      $API_LOG and $UI_LOG"

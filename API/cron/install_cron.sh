#!/usr/bin/env bash
# Install the folder -> API cron job on this machine.
#
#   bash install_cron.sh --inbox /workspace/eval-pipeline/inbox \
#                        --api-url http://localhost:8000 \
#                        --interval-min 10
#
# Writes /etc/cron.d/compnova-eval and (re)starts the cron daemon. Safe to
# re-run: the file is rewritten and cron reloaded.
set -euo pipefail

INBOX=""
API_URL="http://localhost:8000"
INTERVAL_MIN=10
LOG_FILE=""
API_KEY=""
ITEMS="pairs"
TARGET_LANG="spa"
MODE="job"
PIPELINE_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --inbox)        INBOX="$2"; shift 2 ;;
    --api-url)      API_URL="$2"; shift 2 ;;
    --interval-min) INTERVAL_MIN="$2"; shift 2 ;;
    --log)          LOG_FILE="$2"; shift 2 ;;
    --api-key)      API_KEY="$2"; shift 2 ;;
    --items)        ITEMS="$2"; shift 2 ;;
    --target-lang)  TARGET_LANG="$2"; shift 2 ;;
    --mode)         MODE="$2"; shift 2 ;;
    --pipeline-dir) PIPELINE_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

[ -n "$INBOX" ] || { echo "--inbox is required"; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_batch.py"
[ -f "$RUNNER" ] || { echo "run_batch.py not found next to this script"; exit 1; }

PIPELINE_DIR="${PIPELINE_DIR:-$(dirname "$INBOX")}"
LOG_FILE="${LOG_FILE:-/var/log/compnova-eval-batch.log}"
mkdir -p "$(dirname "$LOG_FILE")" "$PIPELINE_DIR"/{inbox,processing,done,failed,results}

PYTHON_BIN="$(command -v python3 || command -v python)"
command -v flock >/dev/null 2>&1 || { echo "flock is required (apt-get install util-linux)"; exit 1; }

# The API key is read from the environment at run time so it never sits in
# the crontab in plain text.
KEY_ENV=""
if [ -n "$API_KEY" ]; then
  KEY_ENV="EVAL_API_KEY='$API_KEY'"
fi

cat > /etc/cron.d/compnova-eval <<EOF
# CompNova evaluation pipeline - installed by API/cron/install_cron.sh
# Picks up audio from $INBOX and evaluates it via $API_URL.
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
$KEY_ENV
*/$INTERVAL_MIN * * * * root /usr/bin/flock -n /run/compnova-eval.lock $PYTHON_BIN "$RUNNER" --inbox "$INBOX" --api-url "$API_URL" --mode $MODE --items $ITEMS --target-lang $TARGET_LANG --require-healthy --log "$LOG_FILE" >> "$LOG_FILE" 2>&1
EOF

chmod 0644 /etc/cron.d/compnova-eval

# cron is not installed in every RunPod image.
if ! command -v cron >/dev/null 2>&1 && ! command -v crond >/dev/null 2>&1; then
  echo "cron is not installed - installing it"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq cron >/dev/null
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache cronie >/dev/null
  fi
fi

if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  systemctl enable cron 2>/dev/null || true
  systemctl restart cron 2>/dev/null || true
elif command -v cron >/dev/null 2>&1; then
  pkill -x cron 2>/dev/null || true
  cron
elif command -v crond >/dev/null 2>&1; then
  pkill -x crond 2>/dev/null || true
  crond
else
  echo "could not start the cron daemon - run the runner from a scheduler instead"
fi

echo
echo "Cron job installed: /etc/cron.d/compnova-eval"
echo "  inbox:    $INBOX"
echo "  api:      $API_URL"
echo "  every:    $INTERVAL_MIN minutes"
echo "  mode:     $MODE   items: $ITEMS   target-lang: $TARGET_LANG"
echo "  log:      $LOG_FILE"
echo
echo "Drop audio into $INBOX and watch it get evaluated:"
echo "  tail -f $LOG_FILE"
echo "  ls $PIPELINE_DIR/results/"

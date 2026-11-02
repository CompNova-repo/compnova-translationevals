#!/usr/bin/env bash
# End-to-end check of the evaluation service on this machine.
#
#   bash API/verify_setup.sh                # against http://localhost:8000
#   bash API/verify_setup.sh --api-url http://<pod>:8000 --api-key KEY
#
# Everything is non-destructive apart from the test evaluations it creates
# (they can be removed afterwards with DELETE /evaluations/{id}).
set -uo pipefail

API_URL="http://localhost:8000"
API_KEY=""
UI_URL="http://localhost:8501"
PIPELINE_DIR="${S2ST_PIPELINE_DIR:-/workspace/eval-pipeline}"

while [ $# -gt 0 ]; do
  case "$1" in
    --api-url)   API_URL="$2"; shift 2 ;;
    --api-key)   API_KEY="$2"; shift 2 ;;
    --ui-url)    UI_URL="$2"; shift 2 ;;
    --pipeline)  PIPELINE_DIR="$2"; shift 2 ;;
    -h|--help)   sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

WORK="$(mktemp -d /tmp/compnova-verify.XXXXXX)"
AUTH=()
[ -n "$API_KEY" ] && AUTH=(-H "X-API-Key: $API_KEY")

PASS=0
FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [ "$1" = "0" ]; then ok "$2"; else bad "$2"; fi; }

echo "== service =="

code="$(curl -s -o /dev/null -w '%{http_code}' "$API_URL/health")"
[ "$code" = "200" ]; check $? "GET /health returns 200 (got $code)"

READY="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "$API_URL/health/ready")"
[ "$READY" = "200" ]
check $? "GET /health/ready returns 200 - the models are loaded (got $READY)"
if [ "$READY" != "200" ]; then
  echo "        model status:"
  curl -s "${AUTH[@]}" "$API_URL/health/ready" | python3 -m json.tool 2>/dev/null | head -40
fi

code="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "$API_URL/docs")"
[ "$code" = "200" ]; check $? "GET /docs (OpenAPI UI) returns 200 (got $code)"

code="$(curl -s -o /dev/null -w '%{http_code}' "$UI_URL")"
[ "$code" = "200" ]; check $? "the dashboard answers on $UI_URL (got $code)"

echo
echo "== storage =="
DB_PATH="$(curl -s "${AUTH[@]}" "$API_URL/health/ready" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("config",{}).get("db_path",""))' 2>/dev/null)"
echo "        database: ${DB_PATH:-unknown}"
if [ -n "$DB_PATH" ] && [ -e "$(dirname "$DB_PATH")" ]; then
  if [ -w "$(dirname "$DB_PATH")" ]; then ok "the database directory is writable"
  else bad "the database directory is NOT writable - evaluations will fail"; fi
else
  bad "the database directory does not exist: ${DB_PATH:-unknown}"
fi

echo
echo "== test audio =="
python3 - "$WORK" <<'PY'
import os, struct, sys
work = sys.argv[1]

def wav(path, seconds=3, rate=16000):
    n = rate * seconds
    data = b"\x00\x00" * n
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate*2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    open(path, "wb").write(hdr + data)

wav(os.path.join(work, "EN_test.wav"))
wav(os.path.join(work, "ES_test.wav"))
print("        wrote silent WAVs (no TTS available for a speech test)")
PY

# Prefer real speech when gTTS is installed - it exercises the whole pipeline
# instead of just the plumbing.
if python3 -c "import gtts" 2>/dev/null; then
  python3 - "$WORK" <<'PY' && echo "        wrote real speech with gTTS"
import os, sys
from gtts import gTTS
work = sys.argv[1]
try:
    gTTS(text="Hello, this is a test of the translation evaluation pipeline.",
         lang="en").save(os.path.join(work, "EN_test.wav"))
    gTTS(text="Hola, esta es una prueba del sistema de evaluacion de traduccion.",
         lang="es").save(os.path.join(work, "ES_test.wav"))
except Exception as exc:
    print(f"        gTTS failed ({exc}) - falling back to silence", file=sys.stderr)
    raise SystemExit(1)
PY
fi

echo
echo "== evaluate (synchronous) =="
RESP="$(curl -s -X POST "${AUTH[@]}" \
  -F "source_audio=@$WORK/EN_test.wav" \
  -F "target_audio=@$WORK/ES_test.wav" \
  "$API_URL/evaluate")"
echo "$RESP" > "$WORK/eval.json"

python3 - "$WORK/eval.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
if "error" in body:
    print("  FAIL  /evaluate returned an error:", body["error"].get("code"), "-", body["error"].get("message")[:120])
    sys.exit(1)
summary = body.get("summary", {})
print(f"        chunks={summary.get('chunks')} comet={summary.get('comet_kiwi')} "
      f"metricx={summary.get('metricx_24')} sentiment={summary.get('sentiment_agreement_pct')}")
if summary.get("chunks", 0) > 0:
    print("  PASS  /evaluate returned scored chunks")
else:
    print("  FAIL  /evaluate returned no chunks")
    sys.exit(1)
if body.get("id"):
    print("  PASS  /evaluate returned an id:", body["id"])
    open(sys.argv[1] + ".id", "w").write(body["id"])
PY
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo
echo "== job API (submit + poll) =="
JOB="$(curl -s -X POST "${AUTH[@]}" \
  -F "source_audio=@$WORK/EN_test.wav" \
  -F "target_audio=@$WORK/ES_test.wav" \
  "$API_URL/jobs")"
JOB_ID="$(echo "$JOB" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("job_id",""))' 2>/dev/null)"
[ -n "$JOB_ID" ]
check $? "POST /jobs returned a job_id ($JOB_ID)"

if [ -n "$JOB_ID" ]; then
  for _ in $(seq 1 180); do
    STATUS="$(curl -s "${AUTH[@]}" "$API_URL/jobs/$JOB_ID" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status",""))' 2>/dev/null)"
    [ "$STATUS" = "done" ] && break
    [ "$STATUS" = "failed" ] && break
    sleep 5
  done
  [ "$STATUS" = "done" ]
  check $? "the job finished with status 'done' (got '$STATUS')"
fi

echo
echo "== idempotency =="
RID="verify-$(date +%s)"
A="$(curl -s -X POST "${AUTH[@]}" -H "X-Request-Id: $RID" \
  -F "source_audio=@$WORK/EN_test.wav" -F "target_audio=@$WORK/ES_test.wav" "$API_URL/evaluate" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin).get("id",""))')"
B="$(curl -s -X POST "${AUTH[@]}" -H "X-Request-Id: $RID" \
  -F "source_audio=@$WORK/EN_test.wav" -F "target_audio=@$WORK/ES_test.wav" "$API_URL/evaluate" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin).get("id",""))')"
[ -n "$A" ] && [ "$A" = "$B" ]
check $? "the same X-Request-Id returns the same evaluation ($A)"

echo
echo "== cron pipeline =="
if [ -f /etc/cron.d/compnova-eval ]; then
  ok "/etc/cron.d/compnova-eval is installed"
  grep -q "require-healthy" /etc/cron.d/compnova-eval
  check $? "the cron job is gated on API readiness"
else
  bad "the cron job is not installed - run: bash API/runpod_start.sh --install-cron"
fi

RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cron/run_batch.py"
if [ -f "$RUNNER" ]; then
  mkdir -p "$PIPELINE_DIR/inbox"
  cp "$WORK/EN_test.wav" "$PIPELINE_DIR/inbox/EN_verify.wav"
  cp "$WORK/ES_test.wav" "$PIPELINE_DIR/inbox/ES_verify.wav"
  OUT="$(python3 "$RUNNER" --inbox "$PIPELINE_DIR/inbox" --api-url "$API_URL" \
        "${AUTH[@]:+--api-key $API_KEY}" --dry-run 2>&1)"
  echo "$OUT" | grep -q "would evaluate"
  check $? "the batch runner pairs the files correctly"
  rm -f "$PIPELINE_DIR/inbox/EN_verify.wav" "$PIPELINE_DIR/inbox/ES_verify.wav"
else
  bad "cron/run_batch.py not found"
fi

echo
echo "== metrics =="
curl -s "${AUTH[@]}" "$API_URL/metrics" | python3 -c '
import json,sys
m = json.load(sys.stdin)
print("        evaluations_total:", m.get("evaluations_total"),
      " jobs_submitted:", m.get("jobs_submitted"),
      " uptime_s:", m.get("uptime_s"))
' 2>/dev/null && ok "GET /metrics returns counters" || bad "GET /metrics failed"

rm -rf "$WORK"

echo
echo "============================================================"
echo "$PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  echo "Some checks failed - see above. The service logs are in /root/logs/api.log."
  exit 1
fi
echo "Everything works. Drop audio into $PIPELINE_DIR/inbox and the cron job will score it."

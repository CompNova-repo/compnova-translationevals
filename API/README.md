# `API/` — the evaluation service

FastAPI service that exposes the S2ST translation-evaluation engine over HTTP, plus the
automation pipeline that feeds it.

```
api.py          FastAPI app: routes, upload handling, job queue, response shaping
config.py       Every environment variable, read once at import time
errors.py       Structured error codes and the retry policy behind each one
eval_core.py    The pipeline - transcribe -> align -> COMET-Kiwi -> MetricX-24 -> sentiment
                (plus the translation path and the batched MetricX evaluation)
eval_store.py   SQLite persistence for evaluations and jobs (standard library only)
cron/           The folder -> API pipeline: batch runner, cron installer, mock API
Dockerfile      GPU image (CUDA 12.1)
Dockerfile.cpu  CPU image (python:3.10-slim)
constraints.txt The versions that are known to work together - one source of truth
runpod_start.sh One-shot setup + start on a RunPod pod, and the cron installer
verify_setup.sh End-to-end check of the whole system
requirements-api.txt
```

## Documentation

| Document | What it covers |
|---|---|
| **[`API_GUIDE.md`](API_GUIDE.md)** | **Start here.** The programmatic contract: every endpoint, request/response schemas, errors, retries, idempotency, score interpretation, performance, and the folder → cron → API pipeline. |
| **[`ROADMAP.md`](ROADMAP.md)** | What was built in 2.0, and what is still open. |
| [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md) | Bringing the service up on a RunPod pod. |
| [`cron/README.md`](cron/README.md) | The reference batch runner and the cron installer. |

## 60-second version

```bash
# start the service (RunPod pod or any Linux GPU box)
bash runpod_start.sh

# install the automation pipeline: audio in a folder -> evaluated
bash runpod_start.sh --install-cron

# prove the whole thing works
bash verify_setup.sh

# evaluate one pair by hand
curl -s -X POST http://localhost:8000/evaluate \
  -F "source_audio=@/data/EN_001.wav" \
  -F "target_audio=@/data/ES_001.wav" | python3 -m json.tool

# or queue it and poll (safe behind a proxy)
curl -s -X POST http://localhost:8000/jobs \
  -F "source_audio=@/data/EN_001.wav" \
  -F "target_audio=@/data/ES_001.wav"
```

Interactive docs: `http://<host>:8000/docs`.

## Four things to know before you build on this

1. **It evaluates, and it can translate.** `POST /translate` and `POST /jobs` (source
   audio only) cover the translation path. See `API_GUIDE.md` §6.5.
2. **Use the job API from a cron job.** `POST /jobs` returns immediately, so a
   ~100 s proxy timeout can never make you duplicate an evaluation. `POST /evaluate`
   blocks and is meant for interactive use.
3. **There is no authentication until you set `EVAL_API_KEYS`.** Don't expose the
   URL before you do.
4. **Results live in SQLite on the API host.** Mount a volume, or export them with
   `GET /evaluations/export`.

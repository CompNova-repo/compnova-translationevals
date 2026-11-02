# What is still needed — running the evaluation API unattended

Companion to [`API_GUIDE.md`](API_GUIDE.md).

**Status as of version 2.0.0:** the P0 and P1 items from the original roadmap are
implemented. This document records what was done and what is still open.

---

## 0. The short version

**The cron pipeline works today.** On a RunPod pod:

```bash
bash API/runpod_start.sh            # start the service
bash API/runpod_start.sh --install-cron   # install the folder pipeline
bash API/verify_setup.sh            # prove it end to end
```

Then drop audio into `/workspace/eval-pipeline/inbox`. The cron job picks it up,
evaluates it through the API, writes a local ledger, and the dashboard shows the
result.

---

## 1. Done in 2.0

### 1.1 The Docker images are fixed

Both `API/Dockerfile` and `API/Dockerfile.cpu` now copy `eval_store.py`, `config.py`
and `errors.py` — `api.py` imports all three at module level, so the containers died
at startup before. They also gained a `VOLUME` for the persistent store, a
`HEALTHCHECK`, and `datasets` (which `metricx24.predict` imports and which was
missing from the image).

### 1.2 MetricX is concurrency-safe and batched

`metricx_eval_batch` now uses **request-unique** input/output file names and is
serialised by a lock, so two concurrent evaluations can no longer overwrite each
other's files.

More importantly, `evaluate_pairs_batch()` scores the chunks of **many** pairs in a
single `metricx24.predict` call. The job worker takes up to `EVAL_JOB_BATCH_SIZE`
queued jobs and batches them. MetricX loads its model **once per batch** instead of
once per file — for a 50-file folder that is the difference between roughly an hour
of pure model loading and roughly a minute.

### 1.3 The API no longer blocks its own event loop

Every endpoint is a plain `def` now, so Starlette runs it in a threadpool and the
event loop stays free. The service answers `/health` and `/evaluations` **while an
evaluation is running**. Previously the dashboard showed "Can't reach the API"
during every single evaluation, because the endpoint was `async def` and did
blocking work on the event loop.

### 1.4 Model warm-up is no longer fatal, and there is a readiness signal

`warm_up()` never raises: it loads each model in a background thread, records the
result per model, and the service starts serving immediately. `GET /health/ready`
returns `503` with a per-model report until everything is loaded. A failed model
download no longer turns into a restart loop with no diagnostics.

`@app.on_event("startup")` was replaced with the `lifespan` context manager.

### 1.5 The result store is persistent by default

`EVAL_DB_PATH` defaults to an absolute path (`/var/lib/compnova/evaluations.db`) on a
volume, not the relative `evaluations.db` that put the database inside the container
filesystem and lost every evaluation on restart. WAL journaling is enabled so a long
listing does not block the writer.

### 1.6 Authentication

`EVAL_API_KEYS` enables `X-API-Key` on every endpoint except the two probes. Multiple
keys are supported so each client can have its own. The service stays open when the
variable is unset — that is deliberate, so a fresh pod works out of the box, and the
start script prints a warning telling you to set it.

### 1.7 Idempotency

`X-Request-Id` makes a request idempotent. A repeat returns the previous result
instead of re-running the evaluation. Stored on the row and searchable via
`GET /evaluations?request_id=...`.

### 1.8 Structured errors

Every failure is now `{"error": {"code", "message", "retryable"}}`. `errors.py` maps
common failures (CUDA OOM, timeouts, connection resets) onto stable codes with a
correct retry policy, and the raw exception stays in the server log.

### 1.9 An asynchronous job API

`POST /jobs` → `202` + `job_id`; `GET /jobs/{id}`; `GET /jobs`; `DELETE /jobs/{id}`.
This is what makes the pipeline safe behind the ~100 s RunPod proxy timeout — the
client never blocks on an evaluation, so it never duplicates one.

### 1.10 Pagination, filtering, export, retention

`GET /evaluations` supports `limit`, `offset`, `since`, `until`, `source_name`,
`request_id` and `order`, and returns `{items, total, limit, offset}`.
`GET /evaluations/export` produces JSONL or CSV. `DELETE /evaluations/{id}` and
`POST /evaluations/prune?before=<epoch>` give you retention.

### 1.11 Input validation

Uploads are checked for extension (`415`), size (`413`) and duration (`422`). The
duration check uses `soundfile` with an `ffprobe` fallback.

### 1.12 Translation moved into the API

`POST /translate` exposes the playground's translation path — Whisper → NLLB-200 →
gTTS, or direct SeamlessM4T. The models are loaded on demand, so a deployment that
only evaluates never pays for them. `POST /jobs` accepts source audio alone and
translates it before evaluating.

### 1.13 The dashboard can start evaluations

`API/DemoApp.py` gained a **New evaluation** tab: upload a pair, or upload a source
clip and pick a language, and it evaluates through the API and renders the result.
The root `DemoApp.py` is now identical to it. (Note that PR #7 replaced the old
standalone playground with the read-only monitor; the translation logic it used to
contain now lives in the API.)

### 1.14 Observability

Per-stage timings in every result (`transcribe_align_s`, `comet_s`, `metricx_s`,
`sentiment_s`, `total_s`), `duration_s` on every record, and a `GET /metrics`
counter endpoint.

### 1.15 The deployment story

`runpod_start.sh --install-cron` installs the pipeline as a real cron job (installing
the cron daemon if the image lacks it), and `API/verify_setup.sh` checks the whole
system end to end — readiness, a real evaluation, the job API, idempotency, the cron
entry, the dashboard and the batch runner.

### 1.16 One source of truth for the pins

`API/constraints.txt` holds the versions that are known to work together, and the
Dockerfile, `runpod_start.sh` and `requirements-api.txt` all reference it.

---

## 2. Still open

### 2.1 The API branch is not merged into `main`

`main` now contains the API (PR #7), but this work is on a separate branch. Merge
order matters: land this branch *before* merging anything that touches `API/`.

### 2.2 MetricX is still a subprocess

Batching took the per-file cost away, but every batch still pays one model load
(~30–60 s). The remaining fix is to keep MetricX resident — either a long-lived
worker process, or importing `metricx24` in-process.

**Why it was not done here:** the upstream `predict.py` is research code whose
`Trainer.predict` path is subtle, and re-implementing it in-process would risk
silently changing the scores. Batching is the safe win; in-process is the next one,
and it should be validated against the current numbers on a fixed corpus first.

### 2.3 No model cache in the image

A fresh pod re-downloads ~10 GB of models. Baking them into the image, or using a
pre-warmed volume snapshot, would remove the 10-minute cold start. Needs a decision
about image size.

### 2.4 No webhooks

The job API has to be polled. A `callback_url` field that the service POSTs to when a
job finishes would remove the polling entirely — worth it if the cron job ever moves
off the API host.

### 2.5 No Prometheus format

`GET /metrics` returns JSON. Exposing the same counters in Prometheus text format
would let a pod be monitored by a standard scraper.

### 2.6 No rate limiting

The API key authenticates a client but does not cap how much it can ask for. A
per-key limit would protect the GPU from a runaway client.

### 2.7 No `/v1` URL prefix

Version 2.0 changed two response shapes (the `/evaluations` envelope and the error
body). There is no versioned prefix, so a future breaking change needs either a
prefix or a coordinated cut-over. `X-API-Version` on the response headers would be a
cheap interim.

### 2.8 SER is not in the API

The playground's Speech Emotion Recognition classifier is English-only and was
disabled upstream (`ENABLE_SER = False`). The API's sentiment comparison is text-based.
Adding audio-based emotion scoring needs a multilingual checkpoint first.

---

## 3. Suggested next steps

1. **Land this branch** and re-run `API/verify_setup.sh` on a real pod.
2. **Set `EVAL_API_KEYS`** before sharing the pod URL with anyone.
3. **Decide where results live.** `/var/lib/compnova` is the pod's own disk — fast
   and consistent, but gone when the pod is recreated. Point `EVAL_DB_PATH` at a
   `/workspace` path if you need them to survive, and export regularly with
   `GET /evaluations/export`.
4. **Validate the batch path on a real folder** and compare the timings against a
   single-file run, to confirm the MetricX saving on your hardware.
5. **Keep MetricX resident** (§2.2) if the batch is large enough to justify it.

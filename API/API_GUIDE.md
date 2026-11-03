# CompNova Translation Evaluation API — Programmatic Guide

**Audience:** engineers integrating the evaluation service into a pipeline (cron, CI,
batch runner, back office).
**Version:** 2.1.0

Everything here is verified against the code in this directory
(`api.py`, `config.py`, `errors.py`, `eval_core.py`, `eval_store.py`).

---

## 0. What is new in 2.1

| Change | Where |
|---|---|
| **Speech Emotion Recognition** is part of the evaluation. `ser_match_score` (cosine similarity over the 8-class emotion distribution) plus `source_emotion_top1` / `target_emotion_top1`, at the top level and in `summary`. | `POST /evaluate`, `POST /jobs` |
| **Audio is validated on ingest.** Corrupt files are rejected with `422`; non-16 kHz / multi-channel audio is resampled to 16 kHz mono. | every upload |
| **Every route is also served at `/api/v1/…`**, and every response carries an `X-API-Version` header. | all routes |
| **`Authorization: Bearer <key>`** is accepted alongside `X-API-Key`. | all routes except the probes |
| **Poison-pill isolation.** If a batch of jobs fails, each job is retried alone, so one bad file cannot fail its neighbours. | `POST /jobs` |
| **Rolling quality window** — mean COMET / MetricX / SER and latency percentiles over the last `EVAL_METRICS_HISTORY` evaluations. | `GET /metrics` |
| **Prometheus output** at `GET /metrics?format=prometheus` and `GET /metrics/prometheus`. | `GET /metrics` |
| **Automatic retention sweep** removes expired evaluations and finished jobs when `EVAL_RETENTION_DAYS > 0`. | background |
| **Jobs survive a crash.** `fail_stale_jobs_on_startup()` marks jobs left `queued`/`running` by a dead process as `failed` with `SERVICE_RESTARTED`. | startup |
| **Job `state` aliases** — `queued`/`running`/`done` are also exposed as `pending`/`processing`/`completed`. | `GET /jobs` |
| **`INSUFFICIENT_DISK`** — a full spool is no longer reported as `GPU_OUT_OF_MEMORY`. | errors |

> [!warning] The SER checkpoint is English-only
> `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` was trained on English
> speech. Same-language accuracy is reported around 0.93; cross-lingual around 0.46 —
> which is why the Streamlit playground shipped with SER switched off. For a
> cross-lingual pair (EN → ES) the **target**-side emotion is out of distribution, so
> `ser_match_score` is **not** a translation-quality signal. Set `EVAL_SER_ENABLED=0`
> to turn it off, or treat the score as advisory. `/health/ready` reports
> `ser_model_language` so a client can tell.

---

## 1. What the API does — and what it does not

The service wraps the **evaluation engine** (`eval_core.py`) behind HTTP, and can
optionally **translate** audio first.

| | |
|---|---|
| **Input** | a *pair* of audio files: source-language speech + its translated counterpart — or just source audio, and the service translates it |
| **Output** | per-chunk quality scores, transcripts, sentiment agreement, a clip-level summary, and per-stage timings |
| **Storage** | every result is persisted to SQLite on the API host; the response also carries a stable `id` |

The pipeline for one evaluation:

```
source_audio ─┐
              ├─► Faster-Whisper (large-v3-turbo, VAD-filtered)  → sentence segments
target_audio ─┘
                    │
                    ├─► LaBSE embeddings + monotonic dynamic programming
                    │     → aligned chunks (1-1, 1-2, 2-1, 2-2 merges allowed)
                    │
                    ├─► COMET-Kiwi      → semantic adequacy / fluency   (0–1, higher better)
                    ├─► MetricX-24 QE  → error-density proxy           (0–25, lower better)
                    └─► multilingual sentiment → 5-class label per side, agreement %
```

Optional translation step (only when you send source audio on its own):

```
source_audio ──► Whisper ──► NLLB-200 ──► gTTS ──► translated_audio
```

### Two ways to run it

| | `POST /evaluate` | `POST /jobs` |
|---|---|---|
| Returns | the full result | `202` + a `job_id` |
| You then | — | poll `GET /jobs/{id}` |
| Survives a proxy timeout | no | **yes** |
| Use it for | interactive use, the dashboard, same-host calls | **cron jobs, remote clients** |

The RunPod proxy times out at ~100 seconds and returns `524` while the evaluation keeps
running on the server. A naive client that retries on `524` creates duplicate work. The
job API removes the problem entirely — see §10.5.

---

## 2. Base URL and where the service runs

FastAPI + Uvicorn on port **8000**.

| Deployment | Base URL |
|---|---|
| RunPod pod (default) | `https://<pod-id>-8000.proxy.runpod.net` |
| Same host / container | `http://localhost:8000` |
| Local dev | `http://127.0.0.1:8000` |

The pod must expose ports **8000** (API) and **8501** (dashboard) as HTTP ports.
Bring-up: [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md).

### Configuration

Everything is read from the environment once, at import time — see `config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token. Required to download the gated **COMET-Kiwi** model. |
| `EVAL_API_KEYS` | *(empty)* | Comma-separated accepted `X-API-Key` values. **Empty = the service is open.** |
| `EVAL_DB_PATH` | `/var/lib/compnova/evaluations.db` | SQLite file holding every evaluation. Absolute, on a persistent volume. |
| `EVAL_SPOOL_DIR` | `/var/lib/compnova/spool` | Where uploads are staged while they wait to be evaluated. |
| `EVAL_JOB_WORKERS` | `1` | How many evaluations may run at once. Raise to `2` on a 48 GB pod. |
| `EVAL_JOB_BATCH_SIZE` | `8` | How many queued jobs share one MetricX model load. |
| `EVAL_MAX_UPLOAD_MB` | `200` | Largest accepted upload. |
| `EVAL_MAX_AUDIO_SECONDS` | `1800` | Longest accepted clip. |
| `EVAL_TRANSLATE_ENABLED` | `1` | Set to `0` to disable the translation models entirely (they are only loaded on demand anyway). |
| `EVAL_RETENTION_DAYS` | `0` | `0` = keep results forever. Above 0, an hourly sweep removes expired evaluations and finished jobs. |
| `EVAL_SER_ENABLED` | `1` | Speech Emotion Recognition. The checkpoint is **English-only** — see the warning at the top. |
| `EVAL_SER_MODEL_LANGUAGE` | `en` | The language the SER checkpoint was trained on. Reported by `/health/ready`. |
| `EVAL_AUTO_RESAMPLE_16K` | `1` | Resample uploads to 16 kHz mono instead of rejecting them. |
| `EVAL_AUDIO_QUALITY_MODE` | `off` | `off` · `flag` (report noise/speaker/overlap flags with every result) · `filter` (also score only the dominant speaker). See §6.16. |
| `EVAL_AUDIO_QUALITY_DIARIZATION` | `1` | Use pyannote diarization when it is installed. |
| `EVAL_AUDIO_QUALITY_LANGUAGE_ID` | `1` | Use Whisper for language identification of the background speech. |
| `EVAL_AUDIO_QUALITY_SNR_FLOOR_DB` | `10.0` | Below this SNR the recording is flagged `noisy`. |
| `EVAL_AUDIO_QUALITY_MIN_SPEECH_RATIO` | `0.10` | Below this share of speech the recording is flagged `mostly_silence`. |
| `EVAL_AUDIO_QUALITY_MAX_DOMINANT_RATIO` | `0.75` | Below this talk-time share no speaker dominates, so the file is flagged. |
| `EVAL_AUDIO_QUALITY_MAX_OVERLAP_RATIO` | `0.15` | Above this share of the file, the recording is flagged `overlapping_speech`. |
| `EVAL_EXPECTED_SAMPLE_RATE` | `16000` | The sample rate the pipeline expects. |
| `EVAL_EXPECTED_CHANNELS` | `1` | The channel count the pipeline expects. |
| `EVAL_MIN_FREE_DISK_MB` | `2048` | Refuse uploads below this much free space (`INSUFFICIENT_DISK`). |
| `EVAL_METRICS_HISTORY` | `500` | How many evaluations the rolling quality window covers. |
| `EVAL_LOG_FORMAT` | `text` | `json` emits one structured line per completed evaluation. |
| `EVAL_CORS_ORIGINS` | *(empty)* | Extra browser origins. Not needed for server-to-server use. |
| `METRICX_DIR` | auto-detected | Where the MetricX-24 checkout lives. |
| `EVAL_API_URL` | `http://localhost:8000` | Read by the **dashboard** and the **cron runner**, not the API. |
| `EVAL_API_KEY` | *(empty)* | Read by the cron runner and the dashboard. |

`GET /health/ready` reports the effective configuration, minus the secrets.

---

## 3. Quick start

```bash
BASE=http://localhost:8000

# 1. Is it up?
curl -s $BASE/health
# {"status":"ok"}

# 2. Are the models loaded? (503 until they are)
curl -s $BASE/health/ready | python3 -m json.tool

# 3. Evaluate a pair - and wait for the result.
curl -s -X POST "$BASE/evaluate" \
  -F "source_audio=@/data/EN_001.wav" \
  -F "target_audio=@/data/ES_001.wav" \
  -o result.json

# 4. Or submit a job and poll it (this is what a cron job should do).
JOB=$(curl -s -X POST "$BASE/jobs" \
  -F "source_audio=@/data/EN_001.wav" \
  -F "target_audio=@/data/ES_001.wav" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
curl -s "$BASE/jobs/$JOB"
```

Python:

```python
import requests

with open("EN_001.wav", "rb") as src, open("ES_001.wav", "rb") as tgt:
    r = requests.post(
        "http://localhost:8000/evaluate",
        files={
            "source_audio": ("EN_001.wav", src, "audio/wav"),
            "target_audio": ("ES_001.wav", tgt, "audio/wav"),
        },
        params={"min_similarity": 0.5, "max_merge": 2},
        headers={"X-API-Key": "your-key"},       # if EVAL_API_KEYS is set
        timeout=(10, 1800),                      # (connect, read)
    )
r.raise_for_status()
result = r.json()
print(result["id"], result["summary"])
```

---

## 4. Request conventions

* **All bodies are `multipart/form-data`.** There is no JSON request body anywhere.
* **Tuning parameters are query parameters, not form fields** — the single most
  common integration mistake:

  ```
  POST /evaluate?min_similarity=0.5&max_merge=2
  ```

* Field names are exactly `source_audio`, `target_audio`, `audio` — case-sensitive.
* The client's filename is preserved and echoed back as `source_name` / `target_name`.
* Send `X-Request-Id` to make a request **idempotent** — a repeat returns the previous
  result instead of re-running the evaluation. See §7.3.

---

## 5. Authentication

If `EVAL_API_KEYS` is set, every endpoint except `/health` and `/health/ready` requires
the header:

```
X-API-Key: <one of the configured keys>
```

Requests without a valid key get `401` with `{"error": {"code": "UNAUTHORIZED", ...}}`.

**If `EVAL_API_KEYS` is empty the service is open** — anyone with the URL can spend GPU
time and read every stored result, including full transcripts. Set it before sharing the
URL:

```bash
export EVAL_API_KEYS="$(openssl rand -hex 24)"
```

Multiple keys are allowed (comma-separated) so each client can have its own — handy for
revoking one without rotating the others.

The cron runner (`cron/run_batch.py`) and the dashboard both read the key from
`EVAL_API_KEY`.

---

## 6. Endpoint reference

### 6.1 `GET /health`

Liveness probe. `200` as soon as the process is up. No authentication required.

```json
{"status": "ok"}
```

### 6.2 `GET /health/ready`

Readiness probe. `200` once every model is loaded, `503` before that or if a model
failed. No authentication required.

```json
{
  "status": "ready",
  "models": {
    "whisper":   {"loaded": true, "seconds": 12.4, "error": null},
    "comet":     {"loaded": true, "seconds": 8.1,  "error": null},
    "aligner":   {"loaded": true, "seconds": 3.2,  "error": null},
    "sentiment": {"loaded": true, "seconds": 4.0,  "error": null},
    "metricx":   {"loaded": true, "seconds": 0.0,  "error": null}
  },
  "missing": [],
  "config": {"version": "2.0.0", "db_path": "...", "auth_required": true, ...},
  "queue_depth": 0
}
```

**Model loading never blocks startup.** The weights load in a background thread, so
the service answers `/health` immediately — and a model that fails to download is
reported here rather than crashing the process. A `524` on `/health/ready` means
"not ready yet"; the `missing` array says which model is the problem.

### 6.3 `GET /metrics`

A small JSON snapshot of counters: evaluations run and failed, jobs submitted,
completed and failed, batches processed, total and last evaluation duration,
uptime, queue depth, and a per-status job census. No authentication required.

### 6.4 `POST /transcribe`

Transcribe one audio file with Faster-Whisper. Returns the joined transcript.

| | |
|---|---|
| Field | `audio` (file, required) |
| Returns | `{"transcript": "<full text>"}` |

```bash
curl -s -X POST "$BASE/transcribe" -F "audio=@/data/EN_001.wav"
```

### 6.5 `POST /translate`

Translate one audio file. The translation models are loaded on first use.

| Field | Required | Meaning |
|---|---|---|
| `source_audio` | yes | The source audio. |
| `target_lang` | no (default `spa`) | `spa`, `fra`, `deu`, `cmn`, `ita`, `por`, `jpn`, `eng`. |
| `mode` | no (default `cascaded`) | `cascaded` (Whisper→NLLB→gTTS) or `seamless` (SeamlessM4T, one shot). |

| Query | Default | Returns |
|---|---|---|
| `response` | `json` | `json` = metadata; `audio` = the translated audio file (`audio/wav`); `both` = JSON with the audio base64-encoded. |

```bash
# metadata
curl -s -X POST "$BASE/translate" \
  -F "source_audio=@/data/EN_001.wav" -F "target_lang=spa"

# the translated audio itself
curl -s -X POST "$BASE/translate?response=audio" \
  -F "source_audio=@/data/EN_001.wav" -F "target_lang=spa" -o ES_001.wav
```

```json
{
  "text": "Hola a todos",
  "source_text": "Hello everyone",
  "target_lang": "spa",
  "mode": "cascaded",
  "content_type": "audio/wav"
}
```

> gTTS (used by the cascaded path) needs outbound internet access to Google's TTS
> endpoint. On a RunPod pod that works; on an air-gapped host use `mode=seamless`,
> which is fully local.

### 6.6 `POST /evaluate`

Evaluate a source/target pair end to end and wait for the result.

**Multipart fields (required)**

| Field | Description |
|---|---|
| `source_audio` | Source-language audio. |
| `target_audio` | Translated audio to evaluate. |

**Query parameters (optional)**

| Parameter | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `min_similarity` | float | `0.5` | `0.0 – 1.0` | Minimum LaBSE cosine similarity for a chunk to count as aligned. Higher = stricter about omissions. |
| `max_merge` | int | `2` | `1 – 3` | Max consecutive segments to try merging on each side. `2` allows 1-1, 1-2, 2-1 and 2-2 alignments. |

**Headers (optional)**

| Header | Meaning |
|---|---|
| `X-API-Key` | Required when `EVAL_API_KEYS` is set. |
| `X-Request-Id` | Idempotency key. A repeat returns the previous result. |

**Response — `200 OK`**

```json
{
  "id": "3f9a1c7e2b04",
  "created_at": 1762201234.567,
  "request_id": null,
  "duration_s": 63.4,
  "transcripts": {
    "source": "Good morning everyone, welcome to the quarterly review.",
    "target": "Buenos días a todos, bienvenidos a la revisión trimestral."
  },
  "comet_score": 0.8123,
  "metricx_score": 4.321,
  "ser_match_score": 0.92,
  "source_emotion_top1": "neutral",
  "target_emotion_top1": "calm",
  "summary": {
    "chunks": 12,
    "unmatched": 1,
    "comet_kiwi": 0.8123,
    "metricx_24": 4.321,
    "sentiment_agreement_pct": 83.3,
    "ser_match_score": 0.92,
    "source_emotion_top1": "neutral",
    "target_emotion_top1": "calm"
  },
  "ser": {
    "ser_match_score": 0.92,
    "source_emotion_top1": "neutral",
    "target_emotion_top1": "calm",
    "source_distribution": [{"label": "neutral", "score": 0.81}, ...],
    "target_distribution": [{"label": "calm", "score": 0.74}, ...]
  },
  "chunks": [
    {
      "source": "Good morning everyone,",
      "target": "Buenos días a todos,",
      "comet_kiwi": 0.9012,
      "metricx_24": 2.104,
      "source_sentiment": "4 stars",
      "target_sentiment": "4 stars",
      "sentiment_match": true
    }
  ],
  "unmatched": {
    "source": ["one segment present in the source but not in the translation"],
    "target": []
  },
  "timings": {
    "total_s": 63.4,
    "transcribe_align_s": 12.1,
    "comet_s": 8.0,
    "metricx_s": 41.2,
    "sentiment_s": 1.3
  },
  "audio_quality": {
    "source": {"snr_db": 18.2, "speaker_count": 2, "flags": ["multi_speaker"]},
    "target": {"snr_db": 31.0, "speaker_count": 1, "flags": []},
    "flags": ["multi_speaker"],
    "contaminated": true,
    "usable": true,
    "mode": "flag"
  }
}
```

**Field semantics**

| Field | Type | Meaning |
|---|---|---|
| `id` | string | 12-hex-char id. Stable handle for `GET /evaluations/{id}`. |
| `created_at` | float | Unix epoch **seconds** (server clock). |
| `duration_s` | float | Wall-clock seconds for the evaluation. |
| `transcripts.source` / `.target` | string | Full concatenated transcript of each file. |
| `summary.chunks` | int | Number of aligned chunks that were scored. |
| `summary.unmatched` | int | Total segments on **either** side that could not be aligned. Non-zero ⇒ possible omission in the translation. |
| `summary.comet_kiwi` | float \| null | Mean COMET-Kiwi over all chunks. **0–1, higher is better.** |
| `summary.metricx_24` | float \| null | Mean MetricX-24 QE. **0–25, lower is better.** |
| `summary.sentiment_agreement_pct` | float \| null | % of chunks where source and target sentiment labels are identical. |
| `summary.ser_match_score` | float \| null | Cosine similarity over the 8-class emotion distributions (0–1, higher = more alike). `null` when SER is off or failed. |
| `summary.source_emotion_top1` / `.target_emotion_top1` | string \| null | The top-1 emotion label on each side. |
| `comet_score` / `metricx_score` | float \| null | Top-level aliases of `summary.comet_kiwi` / `summary.metricx_24`. **`metricx_score` is the raw 0–25 value** — the dashboard plots `1 - score/25` to put it on COMET's 0–1 axis. |
| `ser` | object | The full SER block: both probability distributions plus the score and the top-1 labels. |
| `chunks[]` | array | One entry per aligned chunk. |
| `chunks[].sentiment_match` | bool \| null | `true` if both labels are present and equal. |
| `unmatched.source` / `.target` | string[] | The segment texts that could not be aligned, verbatim. |
| `timings` | object | Seconds spent in each stage. `metricx_s` is the one to watch. |

> The sentiment labels are `"1 star"` … `"5 stars"` — the labels the loaded model
> (`tabularisai/multilingual-sentiment-analysis`) actually produces. Older docs said
> "Very Negative..Very Positive"; that was a different model.

**Errors**

| Status | `error.code` | When |
|---|---|---|
| `401` | `UNAUTHORIZED` | Missing or wrong `X-API-Key`. |
| `413` | `PAYLOAD_TOO_LARGE` | Upload above `EVAL_MAX_UPLOAD_MB`. |
| `415` | `UNSUPPORTED_AUDIO` | The file extension is not an accepted audio format. |
| `422` | `VALIDATION_ERROR` | Missing field, out-of-range parameter, or a clip longer than `EVAL_MAX_AUDIO_SECONDS`. |
| `503` | `GPU_OUT_OF_MEMORY` | The GPU ran out of memory. **Retryable.** |
| `507` | `INSUFFICIENT_DISK` | The spool or the database volume is full. **Not retryable** — an operator has to free the disk. |
| `503` | `MODEL_UNAVAILABLE` | The models are still loading, or one failed. Check `/health/ready`. |
| `503` | `BUSY` | Transient failure. **Retryable.** |
| `504` | `TIMEOUT` | The evaluation took too long. **Retryable.** |
| `500` | `INTERNAL_ERROR` | Anything else. |

Every error has the same shape:

```json
{
  "error": {
    "code": "GPU_OUT_OF_MEMORY",
    "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
    "retryable": true
  }
}
```

`retryable` is the field to branch on — do not string-match the message.

### 6.7 `POST /jobs` → `202 Accepted`

Queue an evaluation. Returns immediately, so it is safe behind any proxy.

**Fields**

| Field | Required | Meaning |
|---|---|---|
| `source_audio` | yes | Source-language audio. |
| `target_audio` | no | Translated audio. If omitted, the service translates the source using `target_lang`. |
| `target_lang` | no (default `spa`) | Target language, used only when `target_audio` is omitted. |
| `translate_mode` | no (default `cascaded`) | `cascaded` or `seamless`. |

Query parameters and headers are the same as `/evaluate`.

**Response — `202`**

```json
{
  "job_id": "6f1c2b9d4e77",
  "status": "queued",
  "created_at": 1762201234.567,
  "request_id": null,
  "poll_url": "/jobs/6f1c2b9d4e77"
}
```

Jobs are processed **in batches**: the worker takes up to `EVAL_JOB_BATCH_SIZE` queued
jobs and evaluates them together, so MetricX-24 loads its model **once per batch**
rather than once per file. That is the difference between a folder of 50 files taking
an hour of pure overhead and taking a minute.

### 6.8 `GET /jobs/{job_id}`

```json
{
  "job_id": "6f1c2b9d4e77",
  "created_at": 1762201234.567,
  "started_at": 1762201235.100,
  "finished_at": 1762201298.400,
  "status": "done",
  "source_name": "EN_001.wav",
  "target_name": "ES_001.wav",
  "request_id": null,
  "result": { "...the same object POST /evaluate returns..." },
  "error": null
}
```

`status` is one of `queued`, `running`, `done`, `failed`, `cancelled`. When it is
`failed`, `error` holds the structured error object.

### 6.9 `GET /jobs`

List jobs (newest first).

| Parameter | Default | Notes |
|---|---|---|
| `limit` | `50` | 1–500. |
| `offset` | `0` | |
| `status` | — | Filter by status. |

```json
{"items": [...], "counts": {"queued": 0, "running": 1, "done": 42, "failed": 0},
 "queue_depth": 1}
```

### 6.10 `DELETE /jobs/{job_id}`

Cancel a `queued` or `running` job. `409` if it has already finished.

### 6.11 `GET /evaluations`

List stored evaluations, newest first. **Metadata only**, so this stays cheap.

| Parameter | Default | Notes |
|---|---|---|
| `limit` | `20` | 1–500. |
| `offset` | `0` | |
| `since` | — | Unix epoch seconds. |
| `until` | — | Unix epoch seconds. |
| `source_name` | — | Exact filename match. |
| `request_id` | — | The idempotency key it was created with. |
| `order` | `desc` | `asc` or `desc`. |

```json
{
  "items": [
    {"id": "3f9a1c7e2b04", "created_at": 1762201234.567,
     "source_name": "EN_001.wav", "target_name": "ES_001.wav",
     "status": "done", "request_id": null, "tags": null, "duration_s": 63.4}
  ],
  "total": 128,
  "limit": 20,
  "offset": 0
}
```

### 6.12 `GET /evaluations/latest`

The most recent evaluation, **including** the full result. `404` if there are none.

### 6.13 `GET /evaluations/{eval_id}`

One evaluation with its full result. `404` if the id is unknown.

### 6.14 `GET /evaluations/export`

Export stored evaluations.

| Parameter | Default | Values |
|---|---|---|
| `format` | `jsonl` | `jsonl` or `csv` |
| `limit` | `5000` | Up to 100000. |

`csv` has one row per evaluation with the summary columns — convenient for a
spreadsheet or a BI tool. `jsonl` has the full records.

### 6.15 `DELETE /evaluations/{eval_id}` / `POST /evaluations/prune?before=<epoch>`

Delete one evaluation, or every evaluation created before a timestamp. `prune` is how
you implement a retention policy.

### 6.16 `POST /audio/quality`

Measure a recording **before** it is evaluated. This is the answer to "what if the
file has background noise and other people talking in the background?" - you do not
silently score a contaminated file, you find out that it is contaminated.

| Field | Required | Meaning |
|---|---|---|
| `audio` | yes | The recording to inspect. |
| `target_audio` | no | A second file, analysed the same way (pair report). |

```bash
curl -s -X POST "$BASE/audio/quality" -F "audio=@/data/EN_001.wav"
```

```json
{
  "source": {
    "duration_s": 63.4,
    "snr_db": 18.2,
    "noise_floor_db": -46.1,
    "speech_level_db": -27.9,
    "speech_ratio": 0.71,
    "speech_s": 45.0,
    "speaker_count": 2,
    "dominant_speaker_ratio": 0.83,
    "overlap_ratio": 0.06,
    "overlap_s": 2.7,
    "languages": ["eng"],
    "background_languages": ["spa"],
    "background_speech_detected": true,
    "detector": "diarization",
    "flags": ["multi_speaker", "background_speech"],
    "usable": true,
    "contaminated": true
  },
  "flags": ["multi_speaker", "background_speech"],
  "contaminated": true,
  "usable": true
}
```

**Flags**

| Flag | Meaning |
|---|---|
| `noisy` | Signal-to-noise ratio below `EVAL_AUDIO_QUALITY_SNR_FLOOR_DB` (default 10 dB). |
| `mostly_silence` | Less than `EVAL_AUDIO_QUALITY_MIN_SPEECH_RATIO` (default 10%) of the file is speech. |
| `multi_speaker` | More than one speaker was detected. |
| `no_single_dominant_speaker` | No speaker owns enough of the speech (default: less than 75% of the talk time). |
| `overlapping_speech` | More than 15% of the file has two people talking at once. |
| `background_speech` | Speech other than the dominant speaker was detected. |
| `very_short_speech` | Less than one second of speech - the scores will be unstable. |

**How the evaluation pipeline uses it**

`EVAL_AUDIO_QUALITY_MODE` controls what happens during `POST /evaluate` and
`POST /jobs`:

| Mode | Behaviour |
|---|---|
| `off` (default) | No analysis. The behaviour of 2.1 is unchanged. |
| `flag` | Every result carries an `audio_quality` block and the flags. The score itself is unchanged - nothing is hidden. |
| `filter` | The transcript and the emotion model only see the dominant speaker's windows, so a background voice in another language cannot contaminate the score. |

```json
"audio_quality": {
  "source": {"...": "..."},
  "target": {"...": "..."},
  "flags": ["multi_speaker", "background_speech"],
  "contaminated": true,
  "usable": true,
  "mode": "flag"
}
```

The analysis needs no model of its own to run: the core measurements are NumPy.
Diarization (pyannote) and language identification (Whisper) improve the report when
they are installed - see `EVAL_AUDIO_QUALITY_DIARIZATION` and
`EVAL_AUDIO_QUALITY_LANGUAGE_ID`.

### 6.17 Endpoint summary

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | no | Liveness |
| `GET` | `/health/ready` | no | Readiness + model status |
| `GET` | `/metrics` | no | Counters + the rolling quality window |
| `GET` | `/metrics/prometheus` | no | The same counters in Prometheus text format |
| `POST` | `/transcribe` | yes | Transcript only |
| `POST` | `/translate` | yes | Translate audio |
| `POST` | `/evaluate` | yes | Evaluate, blocking |
| `POST` | `/jobs` | yes | Evaluate, queued (`202`) |
| `GET` | `/jobs` · `/jobs/{id}` | yes | Job status |
| `DELETE` | `/jobs/{id}` | yes | Cancel a job |
| `GET` | `/evaluations` · `/evaluations/latest` · `/evaluations/{id}` | yes | Read results |
| `GET` | `/evaluations/export` | yes | Export |
| `DELETE` | `/evaluations/{id}` · `POST /evaluations/prune` | yes | Retention |
| `POST` | `/audio/quality` | yes | Measure a recording before scoring it |

---

## 7. Errors, retries and idempotency

### 7.1 The error envelope

Every failure — validation, auth, GPU, timeout — comes back as:

```json
{"error": {"code": "...", "message": "...", "retryable": true|false}}
```

Branch on `code` for behaviour and `retryable` for retry logic. Never string-match
`message`.

### 7.2 Retry policy

| Situation | What to do |
|---|---|
| `401 UNAUTHORIZED` | **Do not retry.** The key is wrong or missing. |
| `413` / `415` / `422` | **Do not retry.** The request is wrong — wrong field, wrong format, or the clip is too long. |
| `503 MODEL_UNAVAILABLE` | Wait. The models are still loading (`/health/ready` will say when). |
| `503 GPU_OUT_OF_MEMORY` | Retry **once** after a pause, with a shorter clip if it repeats. |
| `503 BUSY` | Retry with exponential backoff. |
| `504 TIMEOUT` | Retry with backoff — but reconcile first if you used `/evaluate` (see §10.5). |
| Network error / read timeout | Retry with backoff. Reconcile first — the evaluation may still have completed server-side. |

### 7.3 Idempotency

Send `X-Request-Id` with a key that identifies the *work*, not the request:

```python
request_id = f"{source_filename}:{target_filename}:{source_mtime}"
```

A repeat of the same key returns the **previous result** instead of re-running the
evaluation — so a retry after a timeout never costs you a second GPU run. The key is
stored with the evaluation and is searchable via `GET /evaluations?request_id=...`.

For jobs, the same applies: a repeat returns the existing job — unless that job
already **failed**, in which case a new one is created so a genuine retry is possible.

---

## 8. Reading the scores

| Metric | Range | Direction | Reading it |
|---|---|---|---|
| **COMET-Kiwi** (`wmt22-cometkiwi-da`) | 0 – 1 | higher is better | Reference-free semantic adequacy + fluency. The dashboard flags chunks below **0.40**. |
| **MetricX-24 QE** | 0 – 25 | **lower** is better | Reference-free error-density proxy. The dashboard plots `1 - score/25` so it lands on COMET's 0–1 axis. |
| **`sentiment_agreement_pct`** | 0 – 100 | higher is better | % of aligned chunks where the 5-class label is identical on both sides. |
| **`ser_match_score`** | 0 – 1 | higher is better | Cosine similarity between the source and target 8-class emotion distributions. **English-only checkpoint** — see the warning in §0. |
| **`summary.unmatched`** | count | lower is better | Segments that could not be aligned. Non-zero is a **red flag for omission**. |

Both quality metrics are **reference-free** — no human reference transcript is needed,
which is what makes unattended batch scoring possible. The trade-off is that they are
proxies. Sample and human-audit a subset before making a final claim; use COMET and
MetricX for triage and ranking.

**`min_similarity` is your recall/precision dial.** Too high ⇒ real content shows up in
`unmatched` (false omissions). Too low ⇒ the aligner pairs unrelated sentences (false
matches, inflated scores).

---

## 9. Performance and capacity

A single evaluation is **not** a fast request. Approximate cost on a single GPU pod:

| Stage | Approx. cost | Notes |
|---|---|---|
| Whisper transcription ×2 | seconds → tens of seconds | scales with audio length |
| LaBSE alignment | ~1 s | |
| COMET-Kiwi | a few seconds | model stays resident |
| **MetricX-24** | **~30–60 s of model loading** | see below |
| Sentiment | ~1 s | model stays resident |

> **MetricX is the big one.** `metricx24.predict` is upstream research code with no
> library entry point, so it runs as a subprocess that **loads its model every time**.
> The service calls it **once per batch** rather than once per file: the job worker
> takes up to `EVAL_JOB_BATCH_SIZE` queued jobs and scores all of their chunks in one
> call. A 50-file folder therefore pays the model load once, not 50 times.
>
> Each call uses **request-unique file names** and is serialised by a lock, so two
> concurrent evaluations can never overwrite each other's input/output files.

**Other capacity notes**

* **`EVAL_JOB_WORKERS` controls concurrency.** The default of `1` is safe: the models
  are large and the pipeline is GPU-bound. On a 48 GB pod you can try `2` — the stack
  needs roughly 14 GB per evaluation, so two fit comfortably in 32 GB.
* **Uvicorn runs a single worker process**, and the endpoints are plain `def`
  functions, so Starlette runs them in a threadpool. The event loop stays free: the
  service answers `/health` and `/evaluations` **while an evaluation is running**.
* **Cold start is ~10 minutes** on a fresh pod: ~10 GB of model downloads plus the
  package install. The service answers `/health` immediately; `/health/ready` reports
  `503` until the models are in.
* **The RunPod proxy times out at ~100 seconds** — §10.5.

**Planning numbers.** A 2-minute clip costs roughly **2–4 minutes wall-clock** on an
A-class GPU. 100 pairs ≈ 3–7 GPU-hours on one pod, and much less if they go through
the batch path in one go.

---

## 10. The automated folder pipeline

The target use case: **audio files land in a folder on a schedule; a cron job picks
them up, calls the API, and files the results** — no human in the loop.

### 10.1 Flow

```
  producer                inbox/                cron job (every N min)
  ────────                ──────                ─────────────────────────
  drops EN_001.wav   ──►  EN_001.wav       ──►  pair EN_001 ↔ ES_001
  drops ES_001.wav   ──►  ES_001.wav             move to processing/
                                                   POST /jobs  (multipart)
                                                   poll GET /jobs/{id}
                                                   write results/<id>.json
                                                   move to done/ or failed/
                                                            │
                                                            ▼
                                                   dashboard picks it up
                                                   via GET /evaluations
```

### 10.2 Directory layout and file naming

```
/workspace/eval-pipeline/
├── inbox/          # the watched folder — the producer drops files here
├── processing/     # claimed by the cron job while in flight
├── done/           # successfully evaluated
├── failed/         # gave up — see failed/<key>.error.txt for why
└── results/        # results/<eval_id>.json — the local ledger
```

Pairing rule (`--pair-mode prefix`, the default):

```
<stem>.<src_lang>.<ext>   +   <stem>.<tgt_lang>.<ext>
   EN_001.wav                  ES_001.wav
```

`--pair-mode dotted` matches `clip_001.EN.wav` + `clip_001.ES.wav` instead. This
matches the convention already used in the repo — `download_dral_subset.py` writes
`EN_<id>` / `ES_<id>`.

If the folder holds **source audio only**, run the runner with `--items sources` and
`--target-lang spa`: the API translates each file and then evaluates it.

### 10.3 The reference implementation

`API/cron/run_batch.py` — standard library plus `requests` only.

```bash
cd /var/minis/shared/compnova-translationevals/API/cron

# see what would be sent, send nothing
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --dry-run

# process everything currently in the inbox, then exit
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --once

# long-running watcher
python3 run_batch.py --inbox /workspace/eval-pipeline/inbox --poll 60
```

| Behaviour | Reason |
|---|---|
| **Claims** a file with `os.replace` into `processing/` | Atomic on POSIX — two concurrent runs can never grab the same file. |
| **Pairs by naming convention** | Unpaired source files are submitted for translation instead of being dropped. |
| **Uses the job API by default** (`--mode job`) | Submit + poll survives any proxy timeout. `--mode sync` blocks on one request instead. |
| **Sends `X-Request-Id`** derived from the file names, sizes and mtimes | A retry reuses the previous job instead of duplicating the work. |
| **Reconciles on timeout** | Matches the API's stored results by filename + timestamp before retrying, so a completed-but-timed-out evaluation is not repeated. |
| **Backs off** | 5 s → 15 s → 45 s, up to `--max-attempts`. Non-retryable errors are not retried. |
| **Returns 401/403 files to the inbox** | An auth failure is a configuration problem, not a bad file. |
| **Writes `results/<eval_id>.json`** | A durable local ledger, independent of the API's SQLite. |
| **Writes `failed/<key>.error.txt`** | Why it failed, for the operator. |
| **Serial by default** | The service processes one evaluation at a time by default. |

### 10.4 Scheduling it

**Install it on the pod** (this is the supported path):

```bash
bash API/runpod_start.sh --install-cron
```

That writes `/etc/cron.d/compnova-eval`, installs the cron daemon if the image does
not have it, and prints the paths. It is safe to re-run.

The generated entry:

```cron
*/10 * * * * root /usr/bin/flock -n /run/compnova-eval.lock \
  /usr/bin/python3 .../API/cron/run_batch.py \
    --inbox /workspace/eval-pipeline/inbox \
    --api-url http://localhost:8000 \
    --mode job --items pairs --target-lang spa \
    --require-healthy --log /var/log/compnova-eval-batch.log
```

`flock -n` guarantees no two runs overlap, on top of the script's own atomic
claiming. `--require-healthy` makes the job exit cleanly when the API is not ready,
so a restarting pod does not produce a wall of failures.

**Manually, or with systemd** — see `cron/README.md`.

### 10.5 Same host vs. remote — the 524 caveat

**Same host as the API (recommended).** `http://localhost:8000`. No proxy, no timeout,
no duplicate-work risk.

**Remote (cron on another host).** The RunPod proxy times out at **~100 seconds** and
returns an HTTP **524**. The evaluation usually *keeps running and succeeds* on the
server — the proxy just stops waiting. A naive client that retries on 524 will:

1. fire a duplicate evaluation,
2. overwrite the MetricX temp files of the one still in flight (→ wrong scores),
3. burn GPU time on every retry.

**Use the job API.** `POST /jobs` returns `202` immediately, so nothing ever waits on
the proxy. The client polls `GET /jobs/{id}` — a cheap GET that is not subject to the
evaluation's duration.

The runner already does this: `--mode job` is the default.

For the legacy `--mode sync`, the runner reconciles after a timeout by matching the
API's stored results on the echoed filenames plus a timestamp window, and adopts the
existing evaluation instead of repeating it.

### 10.6 Verifying the pipeline worked

```bash
# what got evaluated?
ls -la /workspace/eval-pipeline/results/ | tail
python3 -m json.tool /workspace/eval-pipeline/results/<eval_id>.json | head -20

# what failed and why?
cat /workspace/eval-pipeline/failed/*.error.txt

# does the API agree with the local ledger?
curl -s "http://localhost:8000/evaluations?limit=5" | python3 -m json.tool

# service counters
curl -s http://localhost:8000/metrics | python3 -m json.tool
```

The **dashboard** (port 8501) reads the same store via `GET /evaluations`, so anything
the cron evaluates shows up in the UI on the next refresh. It can also start
evaluations itself from the **New evaluation** tab.

---

## 11. Deployment reference

### Docker

```bash
docker build -f API/Dockerfile -t compnova-eval-api .
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_your_token \
  -e EVAL_API_KEYS="$(openssl rand -hex 24)" \
  -v compnova-data:/var/lib/compnova \
  compnova-eval-api
```

The volume at `/var/lib/compnova` holds the SQLite store, the upload spool and the
model cache. **Without it, every container restart loses every evaluation.**

> The image also works on CPU: `docker build -f API/Dockerfile.cpu -t compnova-eval-api-cpu .`
> It is just very slow.

### RunPod (the supported path)

`bash API/runpod_start.sh` installs the stack, downloads the models, starts the API on
8000 and the dashboard on 8501. `bash API/runpod_start.sh --check` reports the status,
`--restart` restarts both services, `--install-cron` installs the pipeline job.

Then `bash API/verify_setup.sh` checks the whole thing end to end — readiness, a real
evaluation, the job API, idempotency, the cron entry and the dashboard.

Full details and troubleshooting in [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md).

---

## 12. Versioning

The app reports `version="2.1.0"`. Every route is served at **both** `/<path>` and
`/api/v1/<path>`, and every response carries an `X-API-Version` header. The un-prefixed
paths are kept for compatibility with earlier clients; new clients should use
`/api/v1`.

**What changed in 2.0** (and what a client might notice)

| Change | Impact |
|---|---|
| Endpoints are `def`, not `async def` | The service no longer goes unresponsive during an evaluation. Pure improvement. |
| `GET /evaluations` returns `{items, total, limit, offset}` instead of a bare list | **Breaking.** Use `body["items"]`. |
| Errors use `{"error": {...}}` instead of `{"detail": "..."}` | **Breaking.** Read `error.code` / `error.retryable`. |
| `POST /evaluate` returns `duration_s` and `timings` | Additive. |
| `POST /jobs` added | Additive. |
| `POST /translate` added | Additive. |
| `X-Request-Id` honoured | Additive. |
| `GET /health/ready`, `GET /metrics`, `GET /metrics/prometheus` added | Additive. |
| `ser_match_score`, `source_emotion_top1`, `target_emotion_top1`, `ser` added | Additive. |
| Every route also served at `/api/v1/…` | Additive. |
| `Authorization: Bearer` accepted alongside `X-API-Key` | Additive. |
| `INSUFFICIENT_DISK` added to the error codes | Additive. |
| The dashboard gained an upload tab | Additive. |

Pin the **git SHA of the API host** rather than the version string, and record it
alongside every result — a result is only meaningful if you know which engine produced
it.

**Compatibility rules you can rely on**

* The `summary` and `chunks` field names are stable (the dashboard parses them).
* New fields may be added to `summary` / `chunks` — **ignore unknown keys**.
* `id` and `created_at` remain on the envelope.
* `sentiment_match` may become `null` more often if the sentiment model changes —
  handle `null`.

---

## 13. Where to go next

* [`ROADMAP.md`](ROADMAP.md) — what is still missing, prioritised.
* [`RUNPOD_SETUP.md`](RUNPOD_SETUP.md) — bringing the service up on RunPod.
* [`cron/README.md`](cron/README.md) — the reference batch runner, in detail.
* [`ARCHITECTURE.md`](../ARCHITECTURE.md) — how the evaluation engine works internally.

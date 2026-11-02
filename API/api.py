"""
FastAPI service exposing CompNova's S2ST translation evaluation engine
programmatically. See eval_core.py for the underlying pipeline.

Run directly:      uvicorn api:app --host 0.0.0.0 --port 8000
Interactive docs:   http://<host>:8000/docs

Design notes for version 2.0 (see API/ROADMAP.md)
-------------------------------------------------
* Every endpoint is a plain ``def``, not ``async def``. The evaluation work
  is synchronous and blocking (Whisper, COMET, and a MetricX subprocess), so
  an ``async def`` endpoint ran it on the event loop thread - which made the
  entire service, including /health, unresponsive for the whole duration of
  an evaluation. Starlette runs ``def`` endpoints in a threadpool, so the
  event loop stays free.
* Authentication via the ``X-API-Key`` header (disabled when EVAL_API_KEYS
  is unset, which preserves the previous behaviour).
* An asynchronous job API: ``POST /jobs`` returns 202 immediately and the
  work happens in the background. This is what lets a client survive the
  ~100 s RunPod proxy timeout without creating duplicate evaluations.
* Idempotency via ``X-Request-Id``: a repeated request returns the previous
  result instead of re-running the evaluation.
* Structured errors with a stable code and an explicit ``retryable`` flag.
* ``/health`` (liveness) and ``/health/ready`` (readiness) are separate.
"""
import hmac
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import config
import errors
import eval_core
import eval_store

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
)
LOG = logging.getLogger("compnova.api")

# --------------------------------------------------------------------------- #
# runtime state
# --------------------------------------------------------------------------- #

_stop_event = threading.Event()
_job_queue: "queue.Queue[dict]" = queue.Queue()
_worker_threads: list[threading.Thread] = []

# How many evaluations may run at the same time. The pipeline is GPU-bound
# and the models are large, so the safe default is one at a time.
_eval_slots = threading.BoundedSemaphore(max(1, config.JOB_WORKERS))

_metrics_lock = threading.Lock()
METRICS = {
    "evaluations_total": 0,
    "evaluations_failed": 0,
    "evaluations_seconds": 0.0,
    "jobs_submitted": 0,
    "jobs_completed": 0,
    "jobs_failed": 0,
    "batches_run": 0,
    "last_batch_size": 0,
    "last_evaluation_seconds": None,
    "started_at": time.time(),
}


def _bump(**deltas) -> None:
    with _metrics_lock:
        for key, delta in deltas.items():
            if key not in METRICS or METRICS[key] is None:
                METRICS[key] = 0
            if key.startswith("last_"):
                # "last_*" counters are gauges, not accumulators.
                METRICS[key] = delta
            elif isinstance(delta, (int, float)) and isinstance(METRICS[key], (int, float)):
                METRICS[key] += delta
            else:
                METRICS[key] = delta


# --------------------------------------------------------------------------- #
# background worker
# --------------------------------------------------------------------------- #

def _process_batch(batch: list[dict]) -> None:
    """Evaluate a batch of queued jobs with one MetricX model load."""
    started = time.perf_counter()
    for job in batch:
        eval_store.update_job(job["job_id"], status="running", started_at=time.time())

    try:
        with _eval_slots:
            items = [
                {"source_path": j["source_path"], "target_path": j["target_path"]}
                for j in batch
            ]
            results = eval_core.evaluate_pairs_batch(
                items,
                min_similarity=batch[0]["min_similarity"],
                max_merge=batch[0]["max_merge"],
            )
    except Exception as exc:
        err = errors.classify(exc)
        LOG.exception("batch of %d job(s) failed", len(batch))
        for job in batch:
            eval_store.update_job(
                job["job_id"],
                status="failed",
                finished_at=time.time(),
                error_json=json.dumps(err.to_dict()["error"]),
            )
            _cleanup_paths(job)
        _bump(jobs_failed=len(batch))
        return

    for job, result in zip(batch, results):
        try:
            eval_id, created_at = eval_store.save_evaluation(
                result,
                source_name=job["source_name"],
                target_name=job["target_name"],
                request_id=job.get("request_id"),
                duration_s=result.get("timings", {}).get("total_s"),
                timings=result.get("timings"),
            )
            payload = dict(result)
            payload["id"] = eval_id
            payload["created_at"] = created_at
            payload["source_name"] = job["source_name"]
            payload["target_name"] = job["target_name"]
            payload["request_id"] = job.get("request_id")
            eval_store.update_job(
                job["job_id"],
                status="done",
                finished_at=time.time(),
                result_json=json.dumps(payload, default=str),
            )
            LOG.info("job %s -> evaluation %s", job["job_id"], eval_id)
            _bump(evaluations_total=1, jobs_completed=1)
        except Exception as exc:
            err = errors.classify(exc)
            LOG.exception("could not persist result for job %s", job["job_id"])
            eval_store.update_job(
                job["job_id"],
                status="failed",
                finished_at=time.time(),
                error_json=json.dumps(err.to_dict()["error"]),
            )
            _bump(evaluations_failed=1, jobs_failed=1)
        finally:
            _cleanup_paths(job)

    _bump(batches_run=1, last_batch_size=len(batch))
    LOG.info(
        "batch of %d finished in %.1fs", len(batch), time.perf_counter() - started
    )


def _cleanup_paths(job: dict) -> None:
    for key in ("source_path", "target_path", "translated_path"):
        path = job.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _worker_loop() -> None:
    while not _stop_event.is_set():
        try:
            first = _job_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        batch = [first]
        # Drain whatever else is already queued so the whole batch shares one
        # MetricX model load.
        while len(batch) < config.JOB_BATCH_SIZE:
            try:
                batch.append(_job_queue.get_nowait())
            except queue.Empty:
                break

        try:
            _process_batch(batch)
        except Exception:  # pragma: no cover - defensive
            LOG.exception("worker crashed on a batch")


# --------------------------------------------------------------------------- #
# lifespan
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    eval_store.init_db()

    # Load the models in the background so the service answers /health
    # immediately, even while 10 GB of weights are downloading. A model that
    # fails to load is reported by /health/ready - it no longer stops the
    # process from starting.
    def _warm():
        try:
            eval_core.warm_up()
        except Exception:
            LOG.exception("model warm-up failed")

    warm_thread = threading.Thread(target=_warm, name="model-warmup", daemon=True)
    warm_thread.start()

    for index in range(max(1, config.JOB_WORKERS)):
        thread = threading.Thread(
            target=_worker_loop, name=f"eval-worker-{index}", daemon=True
        )
        thread.start()
        _worker_threads.append(thread)

    LOG.info("service started: %s", config.describe())
    try:
        yield
    finally:
        _stop_event.set()
        for thread in _worker_threads:
            thread.join(timeout=5)
        LOG.info("service stopped")


app = FastAPI(
    title="CompNova Translation Evaluation API",
    description=(
        "Programmatic access to the S2ST translation evaluation engine: "
        "sentence-level semantic alignment (not position-based), "
        "COMET-Kiwi + MetricX-24 quality scoring per chunk, and 5-class "
        "sentiment comparison. "
        "\n\n"
        "Two ways to use it: `POST /evaluate` blocks until the result is "
        "ready, while `POST /jobs` returns 202 immediately and the work "
        "happens in the background - use the job API from anything that "
        "might sit behind a proxy with a request timeout."
    ),
    version=config.API_VERSION,
    lifespan=lifespan,
)

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Accept the request only if it carries a known key.

    When EVAL_API_KEYS is empty the service stays open, which is the
    historical behaviour. Set EVAL_API_KEYS before sharing the URL.
    """
    if not config.API_KEYS:
        return
    for candidate in config.API_KEYS:
        if hmac.compare_digest(candidate, x_api_key or ""):
            return
    raise errors.unauthorized()


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #

@app.exception_handler(errors.ApiError)
def _api_error_handler(_request, exc: errors.ApiError):
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(Exception)
def _unhandled_error_handler(request, exc: Exception):
    LOG.exception("unhandled error on %s %s", request.method, request.url.path)
    err = errors.classify(exc)
    return JSONResponse(status_code=err.status_code, content=err.to_dict())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _save_upload(upload: UploadFile, field: str) -> str:
    """Write an upload to a temp file, enforcing the size limit on the way."""
    filename = upload.filename or ""
    suffix = os.path.splitext(filename)[-1].lower()
    if suffix not in config.ALLOWED_AUDIO_EXTS:
        raise errors.unsupported_media(
            f"{filename or field!r}: expected one of "
            f"{', '.join(sorted(config.ALLOWED_AUDIO_EXTS))}",
            field=field,
        )

    spool_dir = config.SPOOL_DIR
    os.makedirs(spool_dir, exist_ok=True)
    path = os.path.join(spool_dir, f"upload_{uuid.uuid4().hex}{suffix}")
    written = 0
    with open(path, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                out.close()
                os.remove(path)
                raise errors.payload_too_large(
                    f"{filename or field!r} exceeds the "
                    f"{config.MAX_UPLOAD_MB} MB limit",
                    field=field,
                    limit_mb=config.MAX_UPLOAD_MB,
                )
            out.write(chunk)

    if written == 0:
        os.remove(path)
        raise errors.validation(f"{filename or field!r} is empty", field=field)

    _check_duration(path, filename or field)
    return path


def _check_duration(path: str, label: str) -> None:
    """Refuse clips longer than the configured limit."""
    duration = _audio_duration(path)
    if duration is None:
        return
    if duration > config.MAX_AUDIO_SECONDS:
        raise errors.validation(
            f"{label!r} is {duration:.0f}s long, the limit is "
            f"{config.MAX_AUDIO_SECONDS}s",
            duration_s=round(duration, 1),
            limit_s=config.MAX_AUDIO_SECONDS,
        )


def _audio_duration(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _require_models() -> None:
    if not eval_core.models_ready():
        status = eval_core.model_status()
        raise errors.model_unavailable(
            "the evaluation models are still loading or failed to load",
            missing=status.get("missing"),
        )


def _client_request_id(request_id: Optional[str]) -> Optional[str]:
    if request_id:
        request_id = request_id.strip()[:128]
        return request_id or None
    return None


def _tags_from_form(tags: Optional[str]) -> Optional[dict]:
    if not tags:
        return None
    try:
        parsed = json_loads(tags)
    except Exception:
        raise errors.validation("tags must be a JSON object")
    if not isinstance(parsed, dict):
        raise errors.validation("tags must be a JSON object")
    return parsed


def json_loads(raw: str):
    import json
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    """Liveness probe. Says nothing about whether the models are loaded."""
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    """Readiness probe. 503 until every model is loaded."""
    status = eval_core.model_status()
    body = {
        "status": "ready" if status["ready"] else "not_ready",
        "models": status["models"],
        "missing": status["missing"],
        "config": config.describe(),
        "queue_depth": _job_queue.qsize(),
    }
    return JSONResponse(status_code=200 if status["ready"] else 503, content=body)


@app.get("/metrics")
def metrics():
    """A small JSON snapshot of service counters."""
    with _metrics_lock:
        snapshot = dict(METRICS)
    snapshot["uptime_s"] = round(time.time() - snapshot.get("started_at", time.time()), 1)
    snapshot["queue_depth"] = _job_queue.qsize()
    snapshot["jobs"] = eval_store.job_counts()
    return snapshot


# --------------------------------------------------------------------------- #
# transcription and translation
# --------------------------------------------------------------------------- #

@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(..., description="Audio to transcribe (WAV/MP3)"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Transcribe a single audio file (Faster-Whisper)."""
    path = None
    try:
        path = _save_upload(audio, "audio")
        segments = eval_core.transcribe_segments(path)
        return {"transcript": eval_core.join_segment_text(segments)}
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.post("/translate")
def translate(
    source_audio: UploadFile = File(..., description="Source-language audio"),
    target_lang: str = Form("spa", description="Target language code (spa, fra, deu, cmn, ita, por, jpn, eng)"),
    mode: str = Form("cascaded", description="'cascaded' (Whisper->NLLB->gTTS) or 'seamless'"),
    response: str = Query("json", pattern="^(json|audio|both)$"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Translate one audio file.

    ``?response=json`` (default) returns the translated text and metadata.
    ``?response=audio`` returns the translated audio file itself (audio/wav).
    ``?response=both`` returns JSON with the audio base64-encoded.
    """
    if not config.TRANSLATE_ENABLED:
        raise errors.ApiError(
            "TRANSLATE_DISABLED",
            "translation is disabled on this deployment "
            "(set EVAL_TRANSLATE_ENABLED=1 to enable it)",
            503, False,
        )
    if mode not in ("cascaded", "seamless"):
        raise errors.validation("mode must be 'cascaded' or 'seamless'")

    path = None
    out_path = None
    try:
        path = _save_upload(source_audio, "source_audio")
        result = eval_core.translate_audio(path, target_lang, mode=mode)
        out_path = result["audio_path"]
        if not out_path or not os.path.exists(out_path):
            raise errors.internal("translation produced no audio")

        with open(out_path, "rb") as fh:
            payload = fh.read()

        body = {
            "text": result.get("text"),
            "source_text": result.get("source_text"),
            "target_lang": result.get("target_lang"),
            "mode": result.get("mode"),
            "content_type": "audio/wav",
        }

        if response == "audio":
            return StreamingResponse(
                iter([payload]),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=translated.wav"
                },
            )
        if response == "both":
            import base64

            body["audio_base64"] = base64.b64encode(payload).decode("ascii")
        return JSONResponse(content=body)
    finally:
        for p in (out_path, path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

@app.post("/evaluate")
def evaluate(
    source_audio: UploadFile = File(..., description="Source-language audio (WAV/MP3)"),
    target_audio: UploadFile = File(..., description="Translated audio to evaluate (WAV/MP3)"),
    min_similarity: float = Query(
        config.MIN_SIMILARITY_DEFAULT, ge=0.0, le=1.0,
        description="Minimum semantic similarity for a chunk to count as aligned",
    ),
    max_merge: int = Query(
        config.MAX_MERGE_DEFAULT, ge=1, le=3,
        description="Max consecutive segments to try merging on each side",
    ),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Evaluate a source/translated audio pair end to end and wait for the result.

    For a client that may sit behind a request timeout, prefer POST /jobs.
    """
    request_id = _client_request_id(x_request_id)

    if request_id:
        existing = eval_store.find_by_request_id(request_id)
        if existing:
            existing["idempotent"] = True
            return JSONResponse(content=existing)

    _require_models()

    src_path = mt_path = None
    try:
        src_path = _save_upload(source_audio, "source_audio")
        mt_path = _save_upload(target_audio, "target_audio")
        started = time.perf_counter()
        with _eval_slots:
            result = eval_core.evaluate_pair(
                src_path, mt_path,
                min_similarity=min_similarity, max_merge=max_merge,
            )
        duration = time.perf_counter() - started
        eval_id, created_at = eval_store.save_evaluation(
            result,
            source_name=source_audio.filename,
            target_name=target_audio.filename,
            request_id=request_id,
            duration_s=result.get("timings", {}).get("total_s", duration),
            timings=result.get("timings"),
        )
        _bump(evaluations_total=1, evaluations_seconds=duration,
              last_evaluation_seconds=round(duration, 2))
        return JSONResponse(content={
            "id": eval_id,
            "created_at": created_at,
            "request_id": request_id,
            "duration_s": round(duration, 2),
            **result,
        })
    finally:
        for p in (src_path, mt_path):
            if p and os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #

@app.post("/jobs", status_code=202)
def submit_job(
    source_audio: UploadFile = File(..., description="Source-language audio"),
    target_audio: Optional[UploadFile] = File(None, description="Translated audio"),
    target_lang: str = Form("spa", description="Used when target_audio is not supplied"),
    translate_mode: str = Form("cascaded"),
    min_similarity: float = Query(config.MIN_SIMILARITY_DEFAULT, ge=0.0, le=1.0),
    max_merge: int = Query(config.MAX_MERGE_DEFAULT, ge=1, le=3),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Queue an evaluation. Returns 202 immediately.

    Give either both audio files, or only source_audio plus target_lang -
    in the latter case the service translates the source first.
    """
    request_id = _client_request_id(x_request_id)
    if request_id:
        existing = eval_store.find_job_by_request_id(request_id)
        # A job that already failed or was cancelled must not block a retry -
        # only an in-flight or completed job is reused.
        if existing and existing.get("status") in ("queued", "running", "done"):
            existing["idempotent"] = True
            return JSONResponse(status_code=202, content=existing)

    _require_models()

    src_path = None
    mt_path = None
    translated_path = None
    try:
        src_path = _save_upload(source_audio, "source_audio")
        if target_audio is not None:
            mt_path = _save_upload(target_audio, "target_audio")
        else:
            if not config.TRANSLATE_ENABLED:
                raise errors.validation(
                    "target_audio is required on this deployment "
                    "(translation is disabled)"
                )
            translated = eval_core.translate_audio(
                src_path, target_lang, mode=translate_mode
            )
            mt_path = translated["audio_path"]
            translated_path = mt_path

        job_id, created_at = eval_store.create_job(
            source_name=source_audio.filename,
            target_name=(target_audio.filename if target_audio else f"translated:{target_lang}"),
            request_id=request_id,
        )
        job = {
            "job_id": job_id,
            "source_path": src_path,
            "target_path": mt_path,
            "translated_path": translated_path,
            "source_name": source_audio.filename,
            "target_name": (target_audio.filename if target_audio else f"translated:{target_lang}"),
            "request_id": request_id,
            "min_similarity": min_similarity,
            "max_merge": max_merge,
        }
        _job_queue.put(job)
        _bump(jobs_submitted=1)
        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "request_id": request_id,
            "poll_url": f"/jobs/{job_id}",
        })
    except Exception:
        for p in (src_path, mt_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise


@app.get("/jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    return {
        "items": eval_store.list_jobs(limit=limit, offset=offset, status=status),
        "counts": eval_store.job_counts(),
        "queue_depth": _job_queue.qsize(),
    }


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_job(job_id)
    if record is None:
        raise errors.not_found(f"no job {job_id}")
    return record


@app.delete("/jobs/{job_id}")
def cancel_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_job(job_id)
    if record is None:
        raise errors.not_found(f"no job {job_id}")
    if record["status"] in ("queued", "running"):
        eval_store.update_job(job_id, status="cancelled", finished_at=time.time())
        return {"job_id": job_id, "status": "cancelled"}
    raise errors.conflict(f"job {job_id} is already {record['status']}")


# --------------------------------------------------------------------------- #
# stored evaluations
# --------------------------------------------------------------------------- #

@app.get("/evaluations")
def evaluations(
    limit: int = Query(20, ge=1, le=500),
    offset: int = Query(0, ge=0),
    since: Optional[float] = Query(None, description="Unix epoch seconds"),
    until: Optional[float] = Query(None, description="Unix epoch seconds"),
    source_name: Optional[str] = Query(None),
    request_id: Optional[str] = Query(None),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """List stored evaluations, newest first. Metadata only."""
    items = eval_store.list_evaluations(
        limit=limit, offset=offset, since=since, until=until,
        source_name=source_name, request_id=request_id, order=order,
    )
    return {
        "items": items,
        "total": eval_store.count_evaluations(since=since, until=until),
        "limit": limit,
        "offset": offset,
    }


@app.get("/evaluations/latest")
def latest_evaluation(
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_latest()
    if record is None:
        raise errors.not_found("no evaluations yet")
    return record


@app.get("/evaluations/export")
def export_evaluations(
    limit: int = Query(5000, ge=1, le=100000),
    format: str = Query("jsonl", pattern="^(jsonl|csv)$"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Export stored evaluations as JSONL or CSV."""
    records = list(eval_store.iter_evaluations(limit=limit))

    if format == "jsonl":
        import json

        def generate():
            for record in records:
                yield json.dumps(record, default=str) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "id", "created_at", "source_name", "target_name", "chunks", "unmatched",
        "comet_kiwi", "metricx_24", "sentiment_agreement_pct", "duration_s",
    ])
    for record in records:
        summary = record.get("result", {}).get("summary", {})
        writer.writerow([
            record.get("id"), record.get("created_at"), record.get("source_name"),
            record.get("target_name"), summary.get("chunks"), summary.get("unmatched"),
            summary.get("comet_kiwi"), summary.get("metricx_24"),
            summary.get("sentiment_agreement_pct"), record.get("duration_s"),
        ])
    return StreamingResponse(
        iter([buffer.getvalue()]), media_type="text/csv"
    )


@app.get("/evaluations/{eval_id}")
def evaluation(
    eval_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_evaluation(eval_id)
    if record is None:
        raise errors.not_found(f"no evaluation {eval_id}")
    return record


@app.delete("/evaluations/{eval_id}")
def delete_evaluation(
    eval_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    if not eval_store.delete_evaluation(eval_id):
        raise errors.not_found(f"no evaluation {eval_id}")
    return {"deleted": eval_id}


@app.post("/evaluations/prune")
def prune_evaluations(
    before: float = Query(..., description="Unix epoch seconds"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Delete every evaluation created before a timestamp (retention sweep)."""
    removed = eval_store.delete_before(before)
    return {"deleted": removed, "before": before}

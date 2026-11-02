"""
FastAPI service exposing CompNova's S2ST translation evaluation engine
programmatically. See eval_core.py for the underlying pipeline.

Run directly:      uvicorn api:app --host 0.0.0.0 --port 8000
Interactive docs:   http://<host>:8000/docs

Design notes for version 2.1 (see API/ROADMAP.md & Overview for next steps)
---------------------------------------------------------------------------
* Every endpoint is available at both ``/<path>`` and ``/api/v1/<path>``
  (e.g., ``POST /evaluate`` and ``POST /api/v1/evaluate``), and responses carry
  an ``X-API-Version`` header.
* Authentication accepts either ``X-API-Key: <API_KEY>`` or
  ``Authorization: Bearer <API_KEY>`` (disabled when EVAL_API_KEYS is unset).
* Robust upload validation verifies free disk space, audio header integrity,
  sample rate, channel count, and clip duration, rejecting corrupt files with
  422 and automatically resampling non-16kHz/multi-channel clips to 16 kHz mono.
* Batch worker error isolation: if a batch evaluation fails due to a single
  corrupt or problematic file, the worker automatically falls back to per-job
  evaluation so one poison-pill file never fails healthy jobs in the same batch.
* Centralized telemetry & rolling quality window: ``GET /metrics`` tracks rolling
  COMET-Kiwi, MetricX-24, SER match, and latency percentiles over
  ``EVAL_METRICS_HISTORY`` evaluations and supports ``?format=prometheus``
  (plus ``GET /metrics/prometheus``) for Prometheus/Datadog scraping.
* Automatic retention sweep removes expired evaluations and finished jobs when
  ``EVAL_RETENTION_DAYS > 0``.
"""
from collections import deque
import hmac
import inspect
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

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
# runtime state & rolling telemetry
# --------------------------------------------------------------------------- #

_stop_event = threading.Event()
_started_flag = False
_startup_lock = threading.Lock()
_job_queue: "queue.Queue[dict]" = queue.Queue()
_worker_threads: list[threading.Thread] = []

# How many evaluations may run at the same time.
_eval_slots = threading.BoundedSemaphore(max(1, config.JOB_WORKERS))

_metrics_lock = threading.Lock()
_history_ring: deque[dict] = deque(maxlen=max(1, config.METRICS_HISTORY))
_last_retention_sweep = 0.0

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
                METRICS[key] = delta
            elif isinstance(delta, (int, float)) and isinstance(METRICS[key], (int, float)):
                METRICS[key] += delta
            else:
                METRICS[key] = delta


def _enrich_response_fields(result: dict) -> dict:
    """Ensure top-level score aliases from the API spec are always present."""
    out = dict(result)
    summary = dict(out.get("summary") or {})
    ser = out.get("ser") if isinstance(out.get("ser"), dict) else {}

    comet = out.get("comet_score", summary.get("comet_kiwi"))
    metricx = out.get("metricx_score", summary.get("metricx_24"))
    ser_score = out.get("ser_match_score", summary.get("ser_match_score", ser.get("ser_match_score")))
    src_emo = out.get("source_emotion_top1", summary.get("source_emotion_top1", ser.get("source_emotion_top1")))
    tgt_emo = out.get("target_emotion_top1", summary.get("target_emotion_top1", ser.get("target_emotion_top1")))

    out["comet_score"] = comet
    out["metricx_score"] = metricx
    out["ser_match_score"] = ser_score
    out["source_emotion_top1"] = src_emo
    out["target_emotion_top1"] = tgt_emo

    summary.setdefault("comet_kiwi", comet)
    summary.setdefault("metricx_24", metricx)
    summary.setdefault("ser_match_score", ser_score)
    summary.setdefault("source_emotion_top1", src_emo)
    summary.setdefault("target_emotion_top1", tgt_emo)
    out["summary"] = summary
    return out


def _record_telemetry(eval_id: str, result: dict, duration_s: float,
                      source_name: Optional[str], target_name: Optional[str]) -> None:
    summary = result.get("summary") or {}
    entry = {
        "eval_id": eval_id,
        "timestamp": time.time(),
        "duration_s": round(float(duration_s), 3),
        "comet_kiwi": summary.get("comet_kiwi"),
        "metricx_24": summary.get("metricx_24"),
        "ser_match_score": result.get("ser_match_score", summary.get("ser_match_score")),
        "sentiment_agreement_pct": summary.get("sentiment_agreement_pct"),
        "chunks": summary.get("chunks", 0),
        "unmatched": summary.get("unmatched", 0),
    }
    with _metrics_lock:
        _history_ring.append(entry)

    if config.LOG_FORMAT == "json":
        LOG.info(json.dumps({"event": "evaluation_completed", "source": source_name,
                             "target": target_name, **entry}))


def _rolling_quality_summary() -> dict:
    with _metrics_lock:
        items = list(_history_ring)
    if not items:
        return {
            "window_size": 0,
            "avg_comet_kiwi": None,
            "avg_metricx_24": None,
            "avg_ser_match_score": None,
            "avg_sentiment_agreement_pct": None,
            "latency_p50_s": None,
            "latency_p95_s": None,
        }

    def _mean(key):
        vals = [float(x[key]) for x in items if x.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    durations = sorted(float(x["duration_s"]) for x in items if x.get("duration_s") is not None)
    p50 = durations[len(durations) // 2] if durations else None
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else None
    return {
        "window_size": len(items),
        "avg_comet_kiwi": _mean("comet_kiwi"),
        "avg_metricx_24": _mean("metricx_24"),
        "avg_ser_match_score": _mean("ser_match_score"),
        "avg_sentiment_agreement_pct": _mean("sentiment_agreement_pct"),
        "latency_p50_s": round(p50, 2) if p50 is not None else None,
        "latency_p95_s": round(p95, 2) if p95 is not None else None,
    }


def _maybe_run_retention_sweep() -> None:
    global _last_retention_sweep
    if config.RETENTION_DAYS <= 0:
        return
    now = time.time()
    if now - _last_retention_sweep < 3600:
        return
    _last_retention_sweep = now
    cutoff = now - (config.RETENTION_DAYS * 86400)
    try:
        removed = eval_store.delete_before(cutoff)
        if removed:
            LOG.info("retention sweep removed %d expired evaluation(s)", removed)
    except Exception:
        LOG.exception("retention sweep failed")


# --------------------------------------------------------------------------- #
# background worker
# --------------------------------------------------------------------------- #

def _persist_job_result(job: dict, result: dict, elapsed_s: float) -> None:
    try:
        result = _enrich_response_fields(result)
        duration_s = result.get("timings", {}).get("total_s") or round(elapsed_s, 2)
        eval_id, created_at = eval_store.save_evaluation(
            result,
            source_name=job["source_name"],
            target_name=job["target_name"],
            request_id=job.get("request_id"),
            duration_s=duration_s,
            timings=result.get("timings"),
        )
        payload = dict(result)
        payload["id"] = eval_id
        payload["created_at"] = created_at
        payload["source_name"] = job["source_name"]
        payload["target_name"] = job["target_name"]
        payload["request_id"] = job.get("request_id")
        payload["duration_s"] = round(float(duration_s), 2)
        eval_store.update_job(
            job["job_id"],
            status="done",
            finished_at=time.time(),
            result_json=json.dumps(payload, default=str),
        )
        LOG.info("job %s -> evaluation %s", job["job_id"], eval_id)
        _bump(
            evaluations_total=1,
            jobs_completed=1,
            evaluations_seconds=float(duration_s),
            last_evaluation_seconds=round(float(duration_s), 2),
        )
        _record_telemetry(
            eval_id, payload, float(duration_s),
            job.get("source_name"), job.get("target_name"),
        )
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


def _process_batch(batch: list[dict]) -> None:
    """Evaluate a batch of queued jobs with one MetricX model load.

    Includes poison-pill isolation: if a multi-job batch fails, retries each
    job individually so a single bad file never fails its neighbours.
    """
    active_batch = []
    for job in batch:
        rec = eval_store.get_job(job["job_id"])
        if rec and rec.get("status") == "cancelled":
            _cleanup_paths(job)
            continue
        eval_store.update_job(job["job_id"], status="running", started_at=time.time())
        active_batch.append(job)

    if not active_batch:
        return

    started = time.perf_counter()
    try:
        with _eval_slots:
            items = [
                {"source_path": j["source_path"], "target_path": j["target_path"]}
                for j in active_batch
            ]
            results = eval_core.evaluate_pairs_batch(
                items,
                min_similarity=active_batch[0]["min_similarity"],
                max_merge=active_batch[0]["max_merge"],
            )
    except Exception as batch_exc:
        if len(active_batch) == 1:
            err = errors.classify(batch_exc)
            LOG.exception("job %s failed", active_batch[0]["job_id"])
            eval_store.update_job(
                active_batch[0]["job_id"],
                status="failed",
                finished_at=time.time(),
                error_json=json.dumps(err.to_dict()["error"]),
            )
            _cleanup_paths(active_batch[0])
            _bump(evaluations_failed=1, jobs_failed=1)
            return

        LOG.warning(
            "batch of %d jobs failed (%s); isolating jobs individually",
            len(active_batch), batch_exc,
        )
        for job in active_batch:
            job_t0 = time.perf_counter()
            try:
                with _eval_slots:
                    single_res = eval_core.evaluate_pairs_batch(
                        [{"source_path": job["source_path"], "target_path": job["target_path"]}],
                        min_similarity=job["min_similarity"],
                        max_merge=job["max_merge"],
                    )[0]
                _persist_job_result(job, single_res, time.perf_counter() - job_t0)
            except Exception as single_exc:
                err = errors.classify(single_exc)
                LOG.exception("isolated job %s failed", job["job_id"])
                eval_store.update_job(
                    job["job_id"],
                    status="failed",
                    finished_at=time.time(),
                    error_json=json.dumps(err.to_dict()["error"]),
                )
                _cleanup_paths(job)
                _bump(evaluations_failed=1, jobs_failed=1)
        _bump(batches_run=1, last_batch_size=len(active_batch))
        return

    elapsed = time.perf_counter() - started
    per_job_elapsed = elapsed / max(1, len(active_batch))
    for job, result in zip(active_batch, results):
        _persist_job_result(job, result, per_job_elapsed)

    _bump(batches_run=1, last_batch_size=len(active_batch))
    LOG.info("batch of %d finished in %.1fs", len(active_batch), elapsed)


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
        _maybe_run_retention_sweep()
        try:
            first = _job_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        batch = [first]
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
# startup / shutdown lifecycle
# --------------------------------------------------------------------------- #

def _start_service() -> None:
    global _started_flag
    with _startup_lock:
        if _started_flag:
            return
        _stop_event.clear()
        eval_store.init_db()
        eval_store.fail_stale_jobs_on_startup()
        _maybe_run_retention_sweep()

        warm_done = threading.Event()

        def _warm():
            try:
                eval_core.warm_up()
            except Exception:
                LOG.exception("model warm-up failed")
            finally:
                warm_done.set()

        warm_thread = threading.Thread(target=_warm, name="model-warmup", daemon=True)
        warm_thread.start()
        # Wait briefly (up to 0.25s) so fast/stubbed warmups finish before the
        # first request, while real 10 GB model downloads continue in background.
        warm_done.wait(timeout=0.25)

        _worker_threads.clear()
        for index in range(max(1, config.JOB_WORKERS)):
            thread = threading.Thread(
                target=_worker_loop, name=f"eval-worker-{index}", daemon=True
            )
            thread.start()
            _worker_threads.append(thread)

        _started_flag = True
        LOG.info("service started: %s", config.describe())
        if config.SER_ENABLED:
            LOG.warning(
                "SER is enabled. The checkpoint (%s) is %s-only: for a "
                "cross-lingual pair the target-side emotion is out of distribution "
                "and ser_match_score is NOT a translation-quality signal. "
                "Set EVAL_SER_ENABLED=0 to turn it off.",
                eval_core.SER_MODEL_NAME, config.SER_MODEL_LANGUAGE,
            )


def _stop_service() -> None:
    global _started_flag
    with _startup_lock:
        if not _started_flag:
            return
        _stop_event.set()
        for thread in _worker_threads:
            thread.join(timeout=5)
        _worker_threads.clear()
        _started_flag = False
        LOG.info("service stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_service()
    try:
        yield
    finally:
        _stop_service()


_fastapi_kwargs = dict(
    title="CompNova Translation Evaluation API",
    description=(
        "Programmatic access to the S2ST translation evaluation engine: "
        "sentence-level semantic alignment (not position-based), "
        "COMET-Kiwi + MetricX-24 quality scoring per chunk, Speech Emotion "
        "Recognition (SER) cosine similarity, and 5-class sentiment comparison."
        "\n\n"
        "All endpoints are mounted at both `/` and `/api/v1` (e.g. `POST /evaluate` "
        "and `POST /api/v1/evaluate`)."
    ),
    version=config.API_VERSION,
)
if "lifespan" in inspect.signature(FastAPI.__init__).parameters:
    _fastapi_kwargs["lifespan"] = lifespan

app = FastAPI(**_fastapi_kwargs)
if "lifespan" not in _fastapi_kwargs:
    # add_event_handler was removed in modern FastAPI/Starlette. Only use it
    # as a fallback for versions too old to accept a lifespan handler.
    app.add_event_handler("startup", _start_service)
    app.add_event_handler("shutdown", _stop_service)

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def _version_header_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = config.API_VERSION
    return response


# --------------------------------------------------------------------------- #
# auth (supports both X-API-Key and Authorization: Bearer <API_KEY>)
# --------------------------------------------------------------------------- #

def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Accept the request if it carries a known key via X-API-Key or Bearer token.

    When EVAL_API_KEYS is empty the service stays open, which is the
    historical behaviour. Set EVAL_API_KEYS before sharing the URL.
    """
    if not config.API_KEYS:
        return
    # Accept either header. A client that sends a wrong X-API-Key *and* a
    # valid Bearer token should still get in.
    supplied = [x_api_key or "", _extract_bearer_token(authorization) or ""]
    for candidate in config.API_KEYS:
        for value in supplied:
            if value and hmac.compare_digest(candidate, value):
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
# upload & audio validation helpers
# --------------------------------------------------------------------------- #

def _save_upload(upload: UploadFile, field: str) -> str:
    """Write an upload to a temp file, enforcing disk, size, format, and 16kHz mono validation."""
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
    free_mb = config.free_disk_mb(spool_dir)
    if free_mb < config.MIN_FREE_DISK_MB:
        raise errors.insufficient_disk(
            f"insufficient disk space in spool directory ({free_mb:.0f} MB free, "
            f"minimum {config.MIN_FREE_DISK_MB} MB required)",
            free_mb=round(free_mb, 1),
        )

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

    try:
        audio_meta = eval_core.validate_and_normalize_audio(
            path, label=filename or field
        )
    except ValueError as exc:
        if os.path.exists(path):
            os.remove(path)
        raise errors.validation(str(exc), field=field)

    duration = audio_meta.get("duration_s")
    if duration is not None and duration > config.MAX_AUDIO_SECONDS:
        if os.path.exists(path):
            os.remove(path)
        raise errors.validation(
            f"{filename or field!r} is {duration:.0f}s long, the limit is "
            f"{config.MAX_AUDIO_SECONDS}s",
            field=field,
            duration_s=round(duration, 1),
            limit_s=config.MAX_AUDIO_SECONDS,
        )
    return path


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


# --------------------------------------------------------------------------- #
# routes (mounted at both / and /api/v1)
# --------------------------------------------------------------------------- #

router = APIRouter()


@router.get("/health")
def health():
    """Liveness probe. Says nothing about whether the models are loaded."""
    return {"status": "ok"}


@router.get("/health/ready")
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


def _build_metrics_snapshot() -> dict:
    with _metrics_lock:
        snapshot = dict(METRICS)
    snapshot["uptime_s"] = round(time.time() - snapshot.get("started_at", time.time()), 1)
    snapshot["queue_depth"] = _job_queue.qsize()
    snapshot["jobs"] = eval_store.job_counts()
    snapshot["quality_window"] = _rolling_quality_summary()
    snapshot["free_disk_mb"] = round(config.free_disk_mb(config.SPOOL_DIR), 1)
    return snapshot


def _format_prometheus(snapshot: dict) -> str:
    lines = [
        "# HELP compnova_evaluations_total Total evaluations completed.",
        "# TYPE compnova_evaluations_total counter",
        f"compnova_evaluations_total {snapshot.get('evaluations_total', 0)}",
        "# HELP compnova_evaluations_failed_total Total evaluations failed.",
        "# TYPE compnova_evaluations_failed_total counter",
        f"compnova_evaluations_failed_total {snapshot.get('evaluations_failed', 0)}",
        "# HELP compnova_evaluations_seconds_total Total processing seconds spent in evaluations.",
        "# TYPE compnova_evaluations_seconds_total counter",
        f"compnova_evaluations_seconds_total {snapshot.get('evaluations_seconds', 0.0):.3f}",
        "# HELP compnova_jobs_submitted_total Total background jobs submitted.",
        "# TYPE compnova_jobs_submitted_total counter",
        f"compnova_jobs_submitted_total {snapshot.get('jobs_submitted', 0)}",
        "# HELP compnova_jobs_completed_total Total background jobs completed.",
        "# TYPE compnova_jobs_completed_total counter",
        f"compnova_jobs_completed_total {snapshot.get('jobs_completed', 0)}",
        "# HELP compnova_jobs_failed_total Total background jobs failed.",
        "# TYPE compnova_jobs_failed_total counter",
        f"compnova_jobs_failed_total {snapshot.get('jobs_failed', 0)}",
        "# HELP compnova_queue_depth Current in-memory job queue depth.",
        "# TYPE compnova_queue_depth gauge",
        f"compnova_queue_depth {snapshot.get('queue_depth', 0)}",
        "# HELP compnova_uptime_seconds Service uptime in seconds.",
        "# TYPE compnova_uptime_seconds gauge",
        f"compnova_uptime_seconds {snapshot.get('uptime_s', 0.0)}",
    ]
    qw = snapshot.get("quality_window") or {}
    for field, prom_name, help_text in (
        ("avg_comet_kiwi", "compnova_rolling_comet_kiwi", "Rolling mean COMET-Kiwi score (0-1)."),
        ("avg_metricx_24", "compnova_rolling_metricx_24", "Rolling mean MetricX-24 QE score (0-25)."),
        ("avg_ser_match_score", "compnova_rolling_ser_match_score", "Rolling mean SER cosine similarity."),
        ("latency_p50_s", "compnova_latency_p50_seconds", "Rolling median evaluation latency in seconds."),
        ("latency_p95_s", "compnova_latency_p95_seconds", "Rolling p95 evaluation latency in seconds."),
    ):
        val = qw.get(field)
        if val is not None:
            lines.append(f"# HELP {prom_name} {help_text}")
            lines.append(f"# TYPE {prom_name} gauge")
            lines.append(f"{prom_name} {val}")
    return "\n".join(lines) + "\n"


@router.get("/metrics")
def metrics(format: str = Query("json", pattern="^(json|prometheus)$")):
    """Service counters, rolling translation quality telemetry, and Prometheus export."""
    snapshot = _build_metrics_snapshot()
    if format == "prometheus":
        return PlainTextResponse(_format_prometheus(snapshot), media_type="text/plain; version=0.0.4")
    return snapshot


@router.get("/metrics/prometheus")
def metrics_prometheus():
    """Prometheus text-format scrape endpoint."""
    return PlainTextResponse(
        _format_prometheus(_build_metrics_snapshot()),
        media_type="text/plain; version=0.0.4",
    )


@router.post("/transcribe")
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


@router.post("/translate")
def translate(
    source_audio: UploadFile = File(..., description="Source-language audio"),
    target_lang: str = Form("spa", description="Target language code (spa, fra, deu, cmn, ita, por, jpn, eng)"),
    mode: str = Form("cascaded", description="'cascaded' (Whisper->NLLB->gTTS) or 'seamless'"),
    response: str = Query("json", pattern="^(json|audio|both)$"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Translate one audio file."""
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


@router.post("/evaluate")
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
    """Evaluate a source/translated audio pair end to end and wait for the result."""
    request_id = _client_request_id(x_request_id)

    if request_id:
        existing = eval_store.find_by_request_id(request_id)
        if existing:
            res = _enrich_response_fields(existing.get("result") or {})
            return JSONResponse(content={
                "id": existing["id"],
                "created_at": existing["created_at"],
                "request_id": existing.get("request_id"),
                "duration_s": existing.get("duration_s"),
                "idempotent": True,
                **res,
            })

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
        result = _enrich_response_fields(result)
        duration = time.perf_counter() - started
        eval_id, created_at = eval_store.save_evaluation(
            result,
            source_name=source_audio.filename,
            target_name=target_audio.filename,
            request_id=request_id,
            duration_s=result.get("timings", {}).get("total_s", duration),
            timings=result.get("timings"),
        )
        _bump(
            evaluations_total=1,
            evaluations_seconds=duration,
            last_evaluation_seconds=round(duration, 2),
        )
        _record_telemetry(
            eval_id, result, duration, source_audio.filename, target_audio.filename
        )
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


@router.post("/jobs", status_code=202)
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
    """Queue an evaluation. Returns 202 immediately."""
    request_id = _client_request_id(x_request_id)
    if request_id:
        existing = eval_store.find_job_by_request_id(request_id)
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
            "state": "pending",
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


@router.get("/jobs")
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


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_job(job_id)
    if record is None:
        raise errors.not_found(f"no job {job_id}")
    return record


@router.delete("/jobs/{job_id}")
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
        return {"job_id": job_id, "status": "cancelled", "state": "cancelled"}
    raise errors.conflict(f"job {job_id} is already {record['status']}")


@router.get("/evaluations")
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
        "total": eval_store.count_evaluations(since=since, until=until, source_name=source_name),
        "limit": limit,
        "offset": offset,
    }


@router.get("/evaluations/latest")
def latest_evaluation(
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_latest()
    if record is None:
        raise errors.not_found("no evaluations yet")
    return record


@router.get("/evaluations/export")
def export_evaluations(
    limit: int = Query(5000, ge=1, le=100000),
    format: str = Query("jsonl", pattern="^(jsonl|csv)$"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Export stored evaluations as JSONL or CSV."""
    records = list(eval_store.iter_evaluations(limit=limit))

    if format == "jsonl":
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
        "comet_kiwi", "metricx_24", "ser_match_score", "source_emotion_top1",
        "target_emotion_top1", "sentiment_agreement_pct", "duration_s",
    ])
    for record in records:
        res = record.get("result", {})
        summary = res.get("summary", {})
        writer.writerow([
            record.get("id"), record.get("created_at"), record.get("source_name"),
            record.get("target_name"), summary.get("chunks"), summary.get("unmatched"),
            summary.get("comet_kiwi"), summary.get("metricx_24"),
            res.get("ser_match_score", summary.get("ser_match_score")),
            res.get("source_emotion_top1", summary.get("source_emotion_top1")),
            res.get("target_emotion_top1", summary.get("target_emotion_top1")),
            summary.get("sentiment_agreement_pct"), record.get("duration_s"),
        ])
    return StreamingResponse(
        iter([buffer.getvalue()]), media_type="text/csv"
    )


@router.get("/evaluations/{eval_id}")
def evaluation(
    eval_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    record = eval_store.get_evaluation(eval_id)
    if record is None:
        raise errors.not_found(f"no evaluation {eval_id}")
    return record


@router.delete("/evaluations/{eval_id}")
def delete_evaluation(
    eval_id: str,
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    if not eval_store.delete_evaluation(eval_id):
        raise errors.not_found(f"no evaluation {eval_id}")
    return {"deleted": eval_id}


@router.post("/evaluations/prune")
def prune_evaluations(
    before: float = Query(..., description="Unix epoch seconds"),
    x_api_key: Optional[str] = Header(default=None),
    _auth: None = Depends(require_api_key),
):
    """Delete every evaluation created before a timestamp (retention sweep)."""
    removed = eval_store.delete_before(before)
    return {"deleted": removed, "before": before}


# Mount routes at both root (/) and /api/v1 for full compatibility with both
# API_GUIDE.md and Overview for next steps.pdf.
app.include_router(router)
app.include_router(router, prefix="/api/v1")

"""
Central configuration for the CompNova evaluation service.

Everything is read from the environment exactly once, at import time, so the
rest of the code never has to guess. Every value has a working default, so
the service still starts with no configuration at all - which is what the
RunPod setup script relies on.
"""
import os
import shutil


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _default_db_path() -> str:
    """A writable, absolute location for the SQLite store.

    The old default was the relative path "evaluations.db", which put the
    database inside the container filesystem - every pod restart lost every
    evaluation. Prefer /var/lib/compnova (a volume in the Dockerfile) and
    fall back to the working directory if that is not writable.
    """
    preferred = "/var/lib/compnova/evaluations.db"
    parent = os.path.dirname(preferred)
    try:
        os.makedirs(parent, exist_ok=True)
        if os.access(parent, os.W_OK):
            return preferred
    except OSError:
        pass
    return os.path.abspath("evaluations.db")


# ---------------------------------------------------------------- identity --

API_VERSION = "2.1.0"

# ------------------------------------------------------------------- auth --

# Comma-separated list of accepted API keys (accepted via X-API-Key or
# Authorization: Bearer <key>). EMPTY means the service is OPEN - which is
# the historical behaviour. Set EVAL_API_KEYS before exposing the service.
API_KEYS = tuple(
    k.strip()
    for k in os.environ.get("EVAL_API_KEYS", "").split(",")
    if k.strip()
)

# ------------------------------------------------------------------ store --

DB_PATH = os.environ.get("EVAL_DB_PATH", "").strip() or _default_db_path()

# Keep results for this many days. 0 = keep forever.
RETENTION_DAYS = _env_int("EVAL_RETENTION_DAYS", 0)

# --------------------------------------------------------------- requests --

# Where uploaded files are staged while they wait to be evaluated. Must be on
# a volume that survives a restart if jobs are meant to survive one.
SPOOL_DIR = os.environ.get("EVAL_SPOOL_DIR", "").strip() or os.path.join(
    os.path.dirname(_default_db_path()), "spool"
)

# Largest upload accepted, in megabytes.
MAX_UPLOAD_MB = _env_int("EVAL_MAX_UPLOAD_MB", 200)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Longest clip accepted, in seconds. Guards the GPU against a runaway file.
MAX_AUDIO_SECONDS = _env_int("EVAL_MAX_AUDIO_SECONDS", 1800)

# Expected pipeline audio format (16 kHz mono waveform). When
# AUTO_RESAMPLE_16K is enabled, valid audio files with other sample rates or
# channel counts are automatically resampled to 16 kHz mono WAV on ingestion.
EXPECTED_SAMPLE_RATE = _env_int("EVAL_EXPECTED_SAMPLE_RATE", 16000)
EXPECTED_CHANNELS = _env_int("EVAL_EXPECTED_CHANNELS", 1)
AUTO_RESAMPLE_16K = _env_bool("EVAL_AUTO_RESAMPLE_16K", True)

# Audio extensions we are willing to try to decode.
ALLOWED_AUDIO_EXTS = frozenset(
    e.strip().lower()
    for e in os.environ.get(
        "EVAL_ALLOWED_EXTS", ".wav,.mp3,.flac,.m4a,.ogg,.opus,.aac,.wma"
    ).split(",")
    if e.strip()
)

# ------------------------------------------------------------------- jobs --

# How many jobs the background worker processes at once. 1 is the safe
# default: the pipeline is GPU-bound and the models are large. On a 48 GB pod
# you can raise this to 2.
JOB_WORKERS = max(1, _env_int("EVAL_JOB_WORKERS", 1))

# How many queued jobs the worker pulls into a single evaluation batch.
# Batching matters because MetricX-24 is executed as a subprocess that has to
# load its model; one batch = one model load for N files.
JOB_BATCH_SIZE = max(1, _env_int("EVAL_JOB_BATCH_SIZE", 8))

# How long the worker waits for a free slot on the queue.
JOB_QUEUE_TIMEOUT_S = _env_float("EVAL_JOB_QUEUE_TIMEOUT_S", 30.0)

# ------------------------------------------------------------- evaluation --

MIN_SIMILARITY_DEFAULT = _env_float("EVAL_MIN_SIMILARITY", 0.5)
MAX_MERGE_DEFAULT = _env_int("EVAL_MAX_MERGE", 2)

# Speech Emotion Recognition (SER) on 16 kHz mono waveforms.
#
# IMPORTANT: the checkpoint is English-only
# (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition). Same-language
# accuracy is reported around 0.93, cross-lingual around 0.46 - which is why
# the Streamlit playground shipped with SER switched off. For a cross-lingual
# pair (e.g. EN -> ES) the target-side emotion is out of distribution, so
# ser_match_score is NOT a quality signal for the translation. Turn it on only
# when the source and target share a language, or treat the score as advisory.
SER_ENABLED = _env_bool("EVAL_SER_ENABLED", True)

# The language the SER checkpoint was trained on. Surfaced by /health/ready so
# a client can tell whether ser_match_score is meaningful for its pair.
SER_MODEL_LANGUAGE = os.environ.get("EVAL_SER_MODEL_LANGUAGE", "en")

# -------------------------------------------------------------- translation --

# The translation models (NLLB-200, SeamlessM4T) are only loaded when
# /translate is actually called, so a deployment that only evaluates never
# pays for them.
TRANSLATE_ENABLED = _env_bool("EVAL_TRANSLATE_ENABLED", True)

# Where the MetricX checkout lives. The service looks in a few well-known
# places if this is not set explicitly.
METRICX_DIR = os.environ.get("METRICX_DIR", "").strip()


def _find_metricx_dir() -> str:
    if METRICX_DIR:
        return os.path.abspath(METRICX_DIR)
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "metricx"),
        os.path.abspath("metricx"),
        "/app/metricx",
        "/opt/metricx",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "metricx24", "predict.py")):
            return path
    # Fall back to the directory holding this file, so the relative lookup in
    # eval_core still resolves.
    return os.path.dirname(os.path.abspath(__file__))


METRICX_PATH = _find_metricx_dir()

# ------------------------------------------------------------------ server --

# Extra origins allowed to call the API from a browser. Empty = same-origin
# only (which is all the dashboard needs).
CORS_ORIGINS = tuple(
    o.strip()
    for o in os.environ.get("EVAL_CORS_ORIGINS", "").split(",")
    if o.strip()
)

# --------------------------------------------------------------- telemetry --

LOG_LEVEL = os.environ.get("EVAL_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("EVAL_LOG_FORMAT", "text").strip().lower()

# Number of evaluations kept in the in-process metrics ring buffer.
METRICS_HISTORY = _env_int("EVAL_METRICS_HISTORY", 500)

# ------------------------------------------------------------------ misc --

HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))

# Disk free check, used to refuse uploads when the pod is about to fill up.
MIN_FREE_DISK_MB = _env_int("EVAL_MIN_FREE_DISK_MB", 2048)


def free_disk_mb(path: str = "/") -> float:
    try:
        target = path if os.path.exists(path) else (os.path.dirname(path) or "/")
        return shutil.disk_usage(target).free / (1024 * 1024)
    except OSError:
        return float("inf")


def describe() -> dict:
    """A snapshot of the effective configuration, for /health and logs.

    Never includes the API keys themselves.
    """
    return {
        "version": API_VERSION,
        "db_path": DB_PATH,
        "spool_dir": SPOOL_DIR,
        "auth_required": bool(API_KEYS),
        "job_workers": JOB_WORKERS,
        "job_batch_size": JOB_BATCH_SIZE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "expected_sample_rate": EXPECTED_SAMPLE_RATE,
        "expected_channels": EXPECTED_CHANNELS,
        "auto_resample_16k": AUTO_RESAMPLE_16K,
        "ser_enabled": SER_ENABLED,
        "ser_model_language": SER_MODEL_LANGUAGE,
        "translate_enabled": TRANSLATE_ENABLED,
        "metricx_dir": METRICX_PATH,
        "retention_days": RETENTION_DAYS,
        "metrics_history": METRICS_HISTORY,
        "log_format": LOG_FORMAT,
    }

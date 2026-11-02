"""
Persistence for evaluation results and jobs (SQLite, standard library only).

api.py saves every evaluation here, so read-only clients such as DemoApp.py
can list and display past evaluations without loading any models.

Design notes
------------
* The database path comes from ``config.DB_PATH``, which defaults to an
  absolute path on a persistent volume. The previous relative default
  ("evaluations.db") put the file inside the container filesystem, so every
  pod restart silently lost every evaluation.
* WAL journaling is enabled, so a long ``/evaluations`` listing does not
  block the writer.
* Every statement is parameterised. Filenames supplied by a client are
  stored as-is and never interpolated into SQL.
* Job status values support both internal canonical names (``queued``,
  ``running``, ``done``, ``failed``, ``cancelled``) and the architectural
  specification aliases (``pending``, ``processing``, ``completed``,
  ``failed``).
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing

import config

DB_PATH = config.DB_PATH
_lock = threading.Lock()

EVALUATIONS = "evaluations"
JOBS = "jobs"

# Map between internal job statuses and the architectural doc status names
STATUS_TO_SPEC = {
    "queued": "pending",
    "running": "processing",
    "done": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}
SPEC_TO_STATUS = {
    "pending": "queued",
    "processing": "running",
    "completed": "done",
    "queued": "queued",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}


def normalise_job_status(status: str | None) -> str | None:
    if not status:
        return None
    return SPEC_TO_STATUS.get(status.strip().lower(), status.strip().lower())


def _connect() -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(DB_PATH))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _to_builtin(obj):
    """Let json.dumps handle numpy scalars and arrays if any slip through."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _dumps(obj) -> str:
    return json.dumps(obj, default=_to_builtin)


def init_db() -> None:
    """Create the schema if it does not exist. Safe to call repeatedly."""
    with _lock, closing(_connect()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evaluations (
                   id          TEXT PRIMARY KEY,
                   created_at  REAL NOT NULL,
                   source_name TEXT,
                   target_name TEXT,
                   result_json TEXT NOT NULL,
                   request_id  TEXT,
                   tags        TEXT,
                   status      TEXT NOT NULL DEFAULT 'done',
                   duration_s  REAL,
                   timings     TEXT
               )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_created "
            "ON evaluations(created_at)"
        )
        # A client-supplied idempotency key. Partial so that NULLs (the
        # common case) are not considered duplicates of each other.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluations_request_id "
            "ON evaluations(request_id) WHERE request_id IS NOT NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_source_name "
            "ON evaluations(source_name)"
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                   job_id      TEXT PRIMARY KEY,
                   created_at  REAL NOT NULL,
                   started_at  REAL,
                   finished_at REAL,
                   status      TEXT NOT NULL,
                   source_name TEXT,
                   target_name TEXT,
                   request_id  TEXT,
                   result_json TEXT,
                   error_json  TEXT
               )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_request_id "
            "ON jobs(request_id) WHERE request_id IS NOT NULL"
        )
        conn.commit()


# --------------------------------------------------------------------------- #
# evaluations
# --------------------------------------------------------------------------- #

def _row_meta(row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source_name": row["source_name"],
        "target_name": row["target_name"],
        "status": row["status"] if "status" in row.keys() else "done",
        "request_id": row["request_id"] if "request_id" in row.keys() else None,
        "tags": _loads(row["tags"]) if "tags" in row.keys() else None,
        "duration_s": row["duration_s"] if "duration_s" in row.keys() else None,
    }


def _loads(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def save_evaluation(
    result: dict,
    source_name: str | None = None,
    target_name: str | None = None,
    request_id: str | None = None,
    tags: dict | None = None,
    duration_s: float | None = None,
    timings: dict | None = None,
) -> tuple[str, float]:
    """Store one evaluation result. Returns (eval_id, created_at)."""
    eval_id = uuid.uuid4().hex[:12]
    created_at = time.time()
    payload = _dumps(result)
    try:
        with _lock, closing(_connect()) as conn:
            conn.execute(
                "INSERT INTO evaluations "
                "(id, created_at, source_name, target_name, result_json, request_id, "
                " tags, status, duration_s, timings) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'done', ?, ?)",
                (
                    eval_id,
                    created_at,
                    source_name,
                    target_name,
                    payload,
                    request_id,
                    _dumps(tags) if tags else None,
                    duration_s,
                    _dumps(timings) if timings else None,
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        # Two concurrent requests used the same idempotency key. The first one
        # won; return its row instead of failing the second.
        if request_id:
            existing = find_by_request_id(request_id)
            if existing:
                return existing["id"], existing["created_at"]
        raise
    return eval_id, created_at


def find_by_request_id(request_id: str) -> dict | None:
    """Look up a previous evaluation by its idempotency key."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE request_id = ?", (request_id,)
        ).fetchone()
    if row is None:
        return None
    return {**_row_meta(row), "result": json.loads(row["result_json"])}


def list_evaluations(
    limit: int = 20,
    offset: int = 0,
    since: float | None = None,
    until: float | None = None,
    source_name: str | None = None,
    request_id: str | None = None,
    order: str = "desc",
) -> list[dict]:
    """Newest first by default. Metadata only - no full results."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    direction = "ASC" if str(order).lower() == "asc" else "DESC"

    where, params = [], []
    if since is not None:
        where.append("created_at >= ?")
        params.append(float(since))
    if until is not None:
        where.append("created_at <= ?")
        params.append(float(until))
    if source_name:
        where.append("source_name = ?")
        params.append(source_name)
    if request_id:
        where.append("request_id = ?")
        params.append(request_id)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    with closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT id, created_at, source_name, target_name, status, request_id, "
            f"tags, duration_s FROM evaluations {clause} "
            f"ORDER BY created_at {direction} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [_row_meta(r) for r in rows]


def count_evaluations(
    since: float | None = None,
    until: float | None = None,
    source_name: str | None = None,
) -> int:
    where, params = [], []
    if since is not None:
        where.append("created_at >= ?")
        params.append(float(since))
    if until is not None:
        where.append("created_at <= ?")
        params.append(float(until))
    if source_name:
        where.append("source_name = ?")
        params.append(source_name)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with closing(_connect()) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM evaluations {clause}", params
        ).fetchone()
    return int(row["n"])


def get_evaluation(eval_id: str) -> dict | None:
    """Full record, or None if the id is unknown."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (eval_id,)
        ).fetchone()
    if row is None:
        return None
    return {**_row_meta(row), "result": json.loads(row["result_json"])}


def get_latest() -> dict | None:
    items = list_evaluations(limit=1)
    return get_evaluation(items[0]["id"]) if items else None


def delete_evaluation(eval_id: str) -> bool:
    with _lock, closing(_connect()) as conn:
        cur = conn.execute("DELETE FROM evaluations WHERE id = ?", (eval_id,))
        conn.commit()
        return cur.rowcount > 0


def delete_before(cutoff: float) -> int:
    """Retention sweep: remove evaluations and finished jobs created before `cutoff`."""
    with _lock, closing(_connect()) as conn:
        cur = conn.execute("DELETE FROM evaluations WHERE created_at < ?", (cutoff,))
        conn.execute(
            "DELETE FROM jobs WHERE created_at < ? AND status IN ('done', 'failed', 'cancelled')",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount


def iter_evaluations(limit: int = 5000):
    """Full records, oldest first - used by the export endpoint."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluations ORDER BY created_at ASC LIMIT ?",
            (int(limit),),
        ).fetchall()
    for row in rows:
        yield {**_row_meta(row), "result": json.loads(row["result_json"])}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #

def create_job(
    source_name: str | None = None,
    target_name: str | None = None,
    request_id: str | None = None,
) -> tuple[str, float]:
    job_id = uuid.uuid4().hex
    created_at = time.time()
    with _lock, closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, created_at, status, source_name, target_name, "
            "request_id) VALUES (?, ?, 'queued', ?, ?, ?)",
            (job_id, created_at, source_name, target_name, request_id),
        )
        conn.commit()
    return job_id, created_at


def update_job(job_id: str, **fields) -> None:
    """Update the mutable columns of a job row."""
    allowed = {
        "status", "started_at", "finished_at", "result_json", "error_json",
        "source_name", "target_name",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "status" in updates and updates["status"]:
        updates["status"] = normalise_job_status(updates["status"])
    if not updates:
        return
    assignments = ", ".join(f"{k} = ?" for k in updates)
    with _lock, closing(_connect()) as conn:
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE job_id = ?",
            (*updates.values(), job_id),
        )
        conn.commit()


def fail_stale_jobs_on_startup() -> int:
    """Mark in-flight jobs left over from a previous crashed process as failed."""
    now = time.time()
    err = json.dumps({
        "code": "SERVICE_RESTARTED",
        "message": "The service restarted while this job was queued or running.",
        "retryable": True,
    })
    with _lock, closing(_connect()) as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?, error_json = ? "
            "WHERE status IN ('queued', 'running', 'pending', 'processing')",
            (now, err),
        )
        conn.commit()
        return cur.rowcount


def get_job(job_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    status = row["status"]
    return {
        "job_id": row["job_id"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "status": status,
        "state": STATUS_TO_SPEC.get(status, status),
        "source_name": row["source_name"],
        "target_name": row["target_name"],
        "request_id": row["request_id"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": json.loads(row["error_json"]) if row["error_json"] else None,
    }


def find_job_by_request_id(request_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT job_id FROM jobs WHERE request_id = ?", (request_id,)
        ).fetchone()
    if row is None:
        return None
    return get_job(row["job_id"])


def list_jobs(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> list[dict]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    where, params = [], []
    if status:
        norm = normalise_job_status(status)
        where.append("status = ?")
        params.append(norm)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT job_id, created_at, started_at, finished_at, status, "
            f"source_name, target_name, request_id FROM jobs {clause} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["state"] = STATUS_TO_SPEC.get(d["status"], d["status"])
        out.append(d)
    return out


def job_counts() -> dict:
    counts = {
        "queued": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0,
        "pending": 0, "processing": 0, "completed": 0,
    }
    with closing(_connect()) as conn:
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ):
            st = row["status"]
            n = row["n"]
            counts[st] = n
            alias = STATUS_TO_SPEC.get(st)
            if alias and alias != st:
                counts[alias] = n
    return counts


def delete_job(job_id: str) -> bool:
    with _lock, closing(_connect()) as conn:
        cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0

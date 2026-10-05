"""Persistence for evaluation results (SQLite, standard library only).

api.py saves every /evaluate result here, so read-only clients such as
DemoApp.py can list and display past evaluations without loading any models.
Set EVAL_DB_PATH to change where the database file lives.
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing

DB_PATH = os.environ.get("EVAL_DB_PATH", "evaluations.db")
_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _to_builtin(obj):
    """Let json.dumps handle numpy scalars and arrays if any slip through."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def init_db():
    with _lock, closing(_connect()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evaluations (
                   id          TEXT PRIMARY KEY,
                   created_at  REAL NOT NULL,
                   source_name TEXT,
                   target_name TEXT,
                   result_json TEXT NOT NULL
               )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_created "
            "ON evaluations(created_at)"
        )
        conn.commit()


def save_evaluation(result, source_name=None, target_name=None):
    """Store one evaluate_pair() result. Returns (eval_id, created_at)."""
    eval_id = uuid.uuid4().hex[:12]
    created_at = time.time()
    payload = json.dumps(result, default=_to_builtin)
    with _lock, closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?)",
            (eval_id, created_at, source_name, target_name, payload),
        )
        conn.commit()
    return eval_id, created_at


def _meta(row):
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source_name": row["source_name"],
        "target_name": row["target_name"],
    }


def list_evaluations(limit=20):
    """Newest first, metadata only (no full results)."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, created_at, source_name, target_name FROM evaluations "
            "ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [_meta(r) for r in rows]


def get_evaluation(eval_id):
    """Full record, or None if the id is unknown."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (eval_id,)
        ).fetchone()
    if row is None:
        return None
    return {**_meta(row), "result": json.loads(row["result_json"])}


def get_latest():
    items = list_evaluations(limit=1)
    return get_evaluation(items[0]["id"]) if items else None

#!/usr/bin/env python3
"""
Mock of the CompNova evaluation API, for testing run_batch.py offline.

Implements the endpoints the batch runner uses:
    GET  /health          GET  /health/ready     GET  /metrics
    POST /jobs            GET  /jobs/{id}         GET  /jobs
    POST /evaluate        GET  /evaluations      GET  /evaluations/{id}

Behaviour that matters for testing:
  * the FIRST /evaluate is deliberately slow, so the client's timeout
    reconciliation path can be exercised;
  * the first /jobs submission is held for SLOW_SECONDS, so submit-and-poll
    can be exercised;
  * X-API-Key is enforced when MOCK_API_KEYS is set;
  * X-Request-Id is honoured, so a repeated request returns the same job.

Usage:  python3 mock_api.py [port] [slow_seconds]
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

EVALS = []
JOBS = {}
LOCK = threading.Lock()
COUNTERS = {"evaluate": 0, "jobs": 0}
SLOW_SECONDS = 6.0
API_KEYS = tuple(
    k.strip() for k in os.environ.get("MOCK_API_KEYS", "").split(",") if k.strip()
)

FAKE_SUMMARY = {
    "chunks": 3, "unmatched": 0, "comet_kiwi": 0.8,
    "metricx_24": 4.0, "sentiment_agreement_pct": 100.0,
}
FAKE_CHUNKS = [
    {"source": "a", "target": "b", "comet_kiwi": 0.8, "metricx_24": 4.0,
     "source_sentiment": "4 stars", "target_sentiment": "4 stars",
     "sentiment_match": True},
]


def fake_result():
    return {
        "transcripts": {"source": "s", "target": "t"},
        "summary": dict(FAKE_SUMMARY),
        "chunks": [dict(c) for c in FAKE_CHUNKS],
        "unmatched": {"source": [], "target": []},
        "timings": {"total_s": 0.5, "metricx_s": 0.3},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not API_KEYS:
            return True
        return self.headers.get("X-API-Key", "") in API_KEYS

    def _unauthorized(self):
        self._send(401, {"error": {"code": "UNAUTHORIZED",
                                   "message": "missing or invalid X-API-Key",
                                   "retryable": False}})

    def do_GET(self):
        path = self.path.split("?")[0]
        query = dict(re.findall(r"(\w+)=([^&]*)", self.path.split("?")[1])
                     if "?" in self.path else [])
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/health/ready":
            return self._send(200, {"status": "ready", "models": {}, "missing": []})
        if path == "/metrics":
            return self._send(200, {"evaluations_total": len(EVALS)})
        if not self._authorized():
            return self._unauthorized()

        if path == "/evaluations":
            with LOCK:
                items = [{"id": e["id"], "created_at": e["created_at"],
                          "source_name": e["source_name"],
                          "target_name": e["target_name"],
                          "status": "done", "request_id": e.get("request_id")}
                         for e in reversed(EVALS)]
            return self._send(200, {"items": items, "total": len(items),
                                    "limit": 20, "offset": 0})
        if path == "/evaluations/export":
            with LOCK:
                lines = "\n".join(json.dumps(e) for e in EVALS)
            return self._send(200, (lines + "\n").encode(), "application/x-ndjson")
        m = re.match(r"^/evaluations/([^/]+)$", path)
        if m:
            with LOCK:
                for e in EVALS:
                    if e["id"] == m.group(1):
                        return self._send(200, {k: e[k] for k in
                                                ("id", "created_at", "source_name",
                                                 "target_name")} |
                                          {"result": e["result"]})
            return self._send(404, {"error": {"code": "NOT_FOUND",
                                              "message": "not found",
                                              "retryable": False}})
        if path == "/jobs":
            with LOCK:
                items = [{"job_id": j["job_id"], "status": j["status"],
                          "created_at": j["created_at"]}
                         for j in JOBS.values()]
            return self._send(200, {"items": items, "counts": {},
                                    "queue_depth": 0})
        m = re.match(r"^/jobs/([^/]+)$", path)
        if m:
            with LOCK:
                job = JOBS.get(m.group(1))
            if job is None:
                return self._send(404, {"error": {"code": "NOT_FOUND",
                                                  "message": "no such job",
                                                  "retryable": False}})
            return self._send(200, job)
        return self._send(404, {"error": {"code": "NOT_FOUND",
                                          "message": "not found",
                                          "retryable": False}})

    def _parse_multipart(self, raw):
        names = re.findall(r'filename="([^"]+)"', raw)
        fields = dict(re.findall(r'name="([^"]+)"\r\n\r\n([^&]*)\r\n', raw))
        return names, fields

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("latin-1")
        path = self.path.split("?")[0]
        names, fields = self._parse_multipart(raw)

        if path == "/transcribe":
            return self._send(200, {"transcript": "mock"})

        if not self._authorized():
            return self._unauthorized()

        if path == "/evaluate":
            src = names[0] if names else "source"
            tgt = names[1] if len(names) > 1 else "target"
            request_id = self.headers.get("X-Request-Id")
            with LOCK:
                for e in EVALS:
                    if request_id and e.get("request_id") == request_id:
                        return self._send(200, {k: e[k] for k in
                                                ("id", "created_at")} | e["result"])
                COUNTERS["evaluate"] += 1
                n = COUNTERS["evaluate"]
            rec = {"id": f"mock{len(EVALS):04d}", "created_at": time.time(),
                   "source_name": src, "target_name": tgt,
                   "request_id": request_id, "result": fake_result()}
            with LOCK:
                EVALS.append(rec)
            if n == 1:
                time.sleep(SLOW_SECONDS)
            return self._send(200, {k: rec[k] for k in ("id", "created_at")}
                              | rec["result"])

        if path == "/jobs":
            src = names[0] if names else "source"
            tgt = names[1] if len(names) > 1 else f"translated:{fields.get('target_lang', 'spa')}"
            request_id = self.headers.get("X-Request-Id")
            with LOCK:
                if request_id:
                    for j in JOBS.values():
                        if j.get("request_id") == request_id:
                            return self._send(202, j)
                job_id = f"job{len(JOBS):04d}"
                job = {"job_id": job_id, "created_at": time.time(),
                       "status": "queued", "source_name": src,
                       "target_name": tgt, "request_id": request_id,
                       "result": None, "error": None}
                JOBS[job_id] = job
                COUNTERS["jobs"] += 1
                n = COUNTERS["jobs"]

            def run():
                with LOCK:
                    job["status"] = "running"
                if n == 1:
                    time.sleep(SLOW_SECONDS)
                result = fake_result()
                with LOCK:
                    job["status"] = "done"
                    job["result"] = {"id": f"mock{len(EVALS):04d}",
                                     "created_at": time.time(), **result}
                    EVALS.append({"id": job["result"]["id"],
                                  "created_at": job["result"]["created_at"],
                                  "source_name": src, "target_name": tgt,
                                  "request_id": request_id, "result": result})

            threading.Thread(target=run, daemon=True).start()
            return self._send(202, {"job_id": job_id, "status": "queued",
                                    "created_at": job["created_at"],
                                    "request_id": request_id,
                                    "poll_url": f"/jobs/{job_id}"})

        return self._send(404, {"error": {"code": "NOT_FOUND",
                                          "message": "not found",
                                          "retryable": False}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    if len(sys.argv) > 2:
        SLOW_SECONDS = float(sys.argv[2])
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()

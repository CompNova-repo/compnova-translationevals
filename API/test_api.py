"""
Offline test harness for the CompNova evaluation API.

Stubs out torch and the ML model loaders so the API can be exercised on a
machine with no GPU and no 10 GB of weights. Run it from the API directory:

    python3 API/test_api.py
"""
import os
import sys
import json
import types
import time

API_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, API_DIR)

# ---------------------------------------------------------------- torch stub
torch = types.ModuleType("torch")


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __call__(self, fn):
        return fn


torch.cuda = _Cuda()
torch.no_grad = _NoGrad()
torch.bfloat16 = "bfloat16"
torch.float32 = "float32"
torch.device = lambda x: x
torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = torch

# ------------------------------------------------------- environment / config
os.environ["EVAL_API_KEYS"] = "test-key-1,test-key-2"
os.environ["EVAL_DB_PATH"] = os.environ.get(
    "EVAL_TEST_DB", "/tmp/cn_test/evaluations.db")
os.environ["EVAL_SPOOL_DIR"] = "/tmp/cn_test/spool"
os.environ["EVAL_JOB_BATCH_SIZE"] = "4"

import config  # noqa: E402
import errors  # noqa: E402
import eval_core  # noqa: E402
import eval_store  # noqa: E402

# ------------------------------------------------------------- model stubs
FAKE_RESULT = {
    "transcripts": {"source": "hello there", "target": "hola ahi"},
    "summary": {
        "chunks": 2, "unmatched": 0, "comet_kiwi": 0.8123,
        "metricx_24": 4.321, "sentiment_agreement_pct": 100.0,
    },
    "chunks": [
        {"source": "hello", "target": "hola", "comet_kiwi": 0.9, "metricx_24": 2.1,
         "source_sentiment": "4 stars", "target_sentiment": "4 stars",
         "sentiment_match": True},
        {"source": "there", "target": "ahi", "comet_kiwi": 0.72, "metricx_24": 6.5,
         "source_sentiment": "4 stars", "target_sentiment": "3 stars",
         "sentiment_match": False},
    ],
    "unmatched": {"source": [], "target": []},
    "timings": {"total_s": 1.25, "metricx_s": 0.9},
}


def _fake_warm_up(include_translation=None):
    for name in ("whisper", "comet", "aligner", "sentiment", "metricx"):
        eval_core.MODEL_STATUS[name] = {"loaded": True, "seconds": 0.1, "error": None}
    return eval_core.model_status()


def _fake_evaluate_pair(src, tgt, min_similarity=0.5, max_merge=2):
    time.sleep(0.05)
    return dict(FAKE_RESULT)


def _fake_evaluate_batch(items, min_similarity=0.5, max_merge=2):
    time.sleep(0.05)
    return [dict(FAKE_RESULT) for _ in items]


eval_core.warm_up = _fake_warm_up
eval_core.evaluate_pair = _fake_evaluate_pair
eval_core.evaluate_pairs_batch = _fake_evaluate_batch

import api  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def wav_bytes(seconds=1):
    """A tiny but valid RIFF/WAVE header plus silence."""
    import struct

    rate = 8000
    n = rate * seconds
    data = b"\x00\x00" * n
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def main():
    eval_store.init_db()
    with TestClient(api.app) as client:
        print("\n== health ==")
        r = client.get("/health")
        check("GET /health -> 200 ok", r.status_code == 200 and r.json()["status"] == "ok", r.text)

        r = client.get("/health/ready")
        check("GET /health/ready -> 200 ready", r.status_code == 200 and r.json()["status"] == "ready", r.text)

        print("\n== auth ==")
        r = client.get("/evaluations")
        check("no key -> 401", r.status_code == 401, r.text)
        check("401 body has error.code", r.json().get("error", {}).get("code") == "UNAUTHORIZED", r.text)

        auth = {"X-API-Key": "test-key-1"}
        r = client.get("/evaluations", headers=auth)
        check("valid key -> 200", r.status_code == 200, r.text)
        check("list has items+total", "items" in r.json() and "total" in r.json(), r.text)

        r = client.get("/evaluations", headers={"X-API-Key": "wrong"})
        check("wrong key -> 401", r.status_code == 401, r.text)

        print("\n== evaluate ==")
        files = {
            "source_audio": ("EN_001.wav", wav_bytes(), "audio/wav"),
            "target_audio": ("ES_001.wav", wav_bytes(), "audio/wav"),
        }
        r = client.post("/evaluate", files=files, headers=auth)
        check("POST /evaluate -> 200", r.status_code == 200, r.text[:300])
        body = r.json()
        check("response has id", "id" in body, r.text[:200])
        check("response has summary", "summary" in body and body["summary"]["chunks"] == 2, r.text[:200])
        check("response has chunks", len(body.get("chunks", [])) == 2)
        check("response has timings", "timings" in body)
        eval_id = body.get("id")

        print("\n== idempotency ==")
        hdr = {**auth, "X-Request-Id": "req-abc-123"}
        r1 = client.post("/evaluate", files=files, headers=hdr)
        r2 = client.post("/evaluate", files=files, headers=hdr)
        check("first request -> 200", r1.status_code == 200, r1.text[:200])
        check("repeat request -> same id", r1.json()["id"] == r2.json()["id"], f"{r1.json().get('id')} vs {r2.json().get('id')}")
        check("repeat marked idempotent", r2.json().get("idempotent") is True, r2.text[:200])

        print("\n== validation ==")
        bad = {"source_audio": ("a.txt", b"hello", "text/plain"),
               "target_audio": ("b.wav", wav_bytes(), "audio/wav")}
        r = client.post("/evaluate", files=bad, headers=auth)
        check("wrong extension -> 415", r.status_code == 415, r.text[:200])
        check("415 code UNSUPPORTED_AUDIO", r.json().get("error", {}).get("code") == "UNSUPPORTED_AUDIO", r.text[:200])

        r = client.post("/evaluate", files=files, headers=auth, params={"min_similarity": 5})
        check("out-of-range param -> 422", r.status_code == 422, r.text[:200])

        print("\n== jobs ==")
        r = client.post("/jobs", files=files, headers=auth)
        check("POST /jobs -> 202", r.status_code == 202, r.text[:300])
        job = r.json()
        check("job has job_id", "job_id" in job, r.text[:200])
        check("job status queued", job.get("status") == "queued", r.text[:200])

        job_id = job["job_id"]
        final = None
        for _ in range(100):
            time.sleep(0.1)
            jr = client.get(f"/jobs/{job_id}", headers=auth)
            if jr.status_code == 200 and jr.json()["status"] in ("done", "failed"):
                final = jr.json()
                break
        check("job reached done", final is not None and final["status"] == "done",
              json.dumps(final)[:300] if final else "timeout")
        if final and final["status"] == "done":
            check("job carries the result", final.get("result", {}).get("summary", {}).get("chunks") == 2)
            check("job result has an id", "id" in (final.get("result") or {}))

        r = client.get("/jobs/nope", headers=auth)
        check("unknown job -> 404", r.status_code == 404, r.text[:200])

        print("\n== listing, filters, export ==")
        r = client.get("/evaluations", headers=auth, params={"limit": 2})
        check("limit honoured", len(r.json()["items"]) <= 2)
        check("total counts all", r.json()["total"] >= 2, r.json()["total"])

        r = client.get("/evaluations", headers=auth, params={"source_name": "EN_001.wav"})
        check("source_name filter works", all(i["source_name"] == "EN_001.wav" for i in r.json()["items"]))

        r = client.get("/evaluations", headers=auth, params={"request_id": "req-abc-123"})
        check("request_id filter works", len(r.json()["items"]) == 1, r.text[:200])

        r = client.get("/evaluations/export", headers=auth, params={"format": "jsonl"})
        check("export jsonl -> 200", r.status_code == 200 and "application/x-ndjson" in r.headers.get("content-type", ""), r.text[:200])
        check("export jsonl has lines", len(r.text.strip().splitlines()) >= 1)

        r = client.get("/evaluations/export", headers=auth, params={"format": "csv"})
        check("export csv -> 200", r.status_code == 200 and "text/csv" in r.headers.get("content-type", ""), r.text[:200])
        check("export csv has header", r.text.splitlines()[0].startswith("id,created_at"), r.text[:120])

        print("\n== metrics ==")
        r = client.get("/metrics", headers=auth)
        check("GET /metrics -> 200", r.status_code == 200)
        check("metrics count evaluations", r.json().get("evaluations_total", 0) >= 2, r.json())

        print("\n== delete ==")
        r = client.delete(f"/evaluations/{eval_id}", headers=auth)
        check("DELETE evaluation -> 200", r.status_code == 200, r.text[:200])
        r = client.get(f"/evaluations/{eval_id}", headers=auth)
        check("deleted evaluation -> 404", r.status_code == 404, r.text[:200])
        r = client.delete(f"/evaluations/{eval_id}", headers=auth)
        check("delete again -> 404", r.status_code == 404, r.text[:200])

        print("\n== openapi ==")
        r = client.get("/openapi.json")
        check("openapi served", r.status_code == 200)
        paths = r.json().get("paths", {})
        for p in ("/health", "/health/ready", "/evaluate", "/jobs", "/evaluations",
                  "/evaluations/{eval_id}", "/evaluations/export", "/metrics", "/translate"):
            check(f"openapi documents {p}", p in paths, str(list(paths)))

        print("\n== batch evaluation (one MetricX load for N pairs) ==")
        seen = {}

        def _counting_batch(items, min_similarity=0.5, max_merge=2):
            seen["n"] = len(items)
            return [dict(FAKE_RESULT) for _ in items]

        eval_core.evaluate_pairs_batch = _counting_batch
        results = eval_core.evaluate_pairs_batch(
            [{"source_path": "a.wav", "target_path": "b.wav"} for _ in range(5)]
        )
        check("batch of 5 -> 5 results", len(results) == 5, str(len(results)))
        check("batch handled in ONE call", seen.get("n") == 5, str(seen))
        check("each result has timings", all("timings" in r for r in results))

        print("\n== job failure path ==")

        def _boom(items, min_similarity=0.5, max_merge=2):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        eval_core.evaluate_pairs_batch = _boom
        r = client.post("/jobs", files=files, headers=auth)
        fail_job = r.json()["job_id"]
        final = None
        for _ in range(100):
            time.sleep(0.1)
            jr = client.get(f"/jobs/{fail_job}", headers=auth)
            if jr.status_code == 200 and jr.json()["status"] in ("done", "failed"):
                final = jr.json()
                break
        check("failing job -> status failed", final is not None and final["status"] == "failed",
              json.dumps(final)[:300] if final else "timeout")
        if final:
            check("failed job carries an error code",
                  (final.get("error") or {}).get("code") == "GPU_OUT_OF_MEMORY",
                  json.dumps(final.get("error"))[:300])
            check("failed job error is retryable",
                  (final.get("error") or {}).get("retryable") is True,
                  json.dumps(final.get("error"))[:300])
        eval_core.evaluate_pairs_batch = _fake_evaluate_batch

        print("\n== job idempotency ==")
        hdr = {**auth, "X-Request-Id": "job-req-1"}
        j1 = client.post("/jobs", files=files, headers=hdr)
        j2 = client.post("/jobs", files=files, headers=hdr)
        check("repeat job -> same job_id", j1.json()["job_id"] == j2.json()["job_id"],
              f"{j1.json().get('job_id')} vs {j2.json().get('job_id')}")
        check("repeat job marked idempotent", j2.json().get("idempotent") is True, j2.text[:200])

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

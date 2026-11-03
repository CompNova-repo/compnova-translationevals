"""
Offline test suite for the CompNova evaluation service.

No GPU, no model weights, no network: `torch` is stubbed, every model loader
is faked, and the evaluation pipeline returns canned results. The point is to
exercise the entire HTTP surface - routing, auth, validation, the job queue,
the store, telemetry and error handling - without loading a single model.

    pip install fastapi httpx numpy python-multipart
    python3 API/test_api.py

For the on-pod check against the real models, see API/verify_setup.sh.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import testkit  # noqa: E402

API, EVAL_CORE, EVAL_STORE, FAKES = testkit.install_everything(
    "/tmp/compnova_test/evaluations.db"
)

from fastapi.testclient import TestClient  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


AUTH = {"X-API-Key": "test-key-1"}
BEARER = {"Authorization": "Bearer test-key-2"}


def wav(seconds=1.0, rate=16000):
    return testkit.wav_bytes(seconds=seconds, rate=rate)


def main():
    with TestClient(API.app) as client:
        # ---------------------------------------------------------- health
        print("\n== health & readiness ==")
        r = client.get("/health")
        check("GET /health -> 200 ok", r.status_code == 200 and r.json()["status"] == "ok", r.text)
        check("X-API-Version header present",
              r.headers.get("x-api-version") == "2.1.0", str(dict(r.headers)))
        r = client.get("/health/ready")
        check("GET /health/ready -> 200 ready",
              r.status_code == 200 and r.json()["status"] == "ready", r.text)
        check("readiness lists the models", "whisper" in r.json().get("models", {}), r.text[:200])
        check("readiness reports the SER model language",
              r.json().get("config", {}).get("ser_model_language") == "en", r.text[:300])
        r = client.get("/api/v1/health")
        check("GET /api/v1/health -> 200 ok", r.status_code == 200, r.text)

        # ------------------------------------------------------------ auth
        print("\n== authentication ==")
        r = client.get("/evaluations")
        check("no key -> 401", r.status_code == 401, r.text)
        check("401 body has error.code",
              r.json().get("error", {}).get("code") == "UNAUTHORIZED", r.text)
        check("401 is not retryable",
              r.json().get("error", {}).get("retryable") is False, r.text)
        r = client.get("/evaluations", headers={"X-API-Key": "wrong"})
        check("wrong key -> 401", r.status_code == 401, r.text)
        r = client.get("/evaluations", headers=AUTH)
        check("valid X-API-Key -> 200", r.status_code == 200, r.text)
        r = client.get("/api/v1/evaluations", headers=BEARER)
        check("valid Authorization: Bearer -> 200", r.status_code == 200, r.text)
        r = client.get("/evaluations", headers={"X-API-Key": "wrong", **BEARER})
        check("wrong X-API-Key + valid Bearer -> 200", r.status_code == 200, r.text)
        r = client.get("/health")
        check("health needs no key", r.status_code == 200, r.text)

        # --------------------------------------------------------- evaluate
        print("\n== evaluate ==")
        files = {
            "source_audio": ("EN_001.wav", wav(), "audio/wav"),
            "target_audio": ("ES_001.wav", wav(), "audio/wav"),
        }
        r = client.post("/evaluate", files=files, headers=AUTH)
        check("POST /evaluate -> 200", r.status_code == 200, r.text[:300])
        body = r.json()
        check("response has id", "id" in body, r.text[:200])
        check("response has summary", body.get("summary", {}).get("chunks") == 2, r.text[:200])
        check("response has chunks", len(body.get("chunks", [])) == 2)
        check("response has timings", "timings" in body)
        check("response has duration_s", "duration_s" in body)
        check("top-level comet_score alias", body.get("comet_score") == 0.8123, r.text[:200])
        check("top-level metricx_score alias", body.get("metricx_score") == 4.321, r.text[:200])
        check("SER score at top level", body.get("ser_match_score") == 0.92, r.text[:200])
        check("SER emotions at top level",
              body.get("source_emotion_top1") == "neutral"
              and body.get("target_emotion_top1") == "calm", r.text[:200])
        check("SER in summary", body.get("summary", {}).get("ser_match_score") == 0.92, r.text[:200])
        check("ser block present", isinstance(body.get("ser"), dict), r.text[:200])
        eval_id = body.get("id")

        r = client.post("/api/v1/evaluate", files=files, headers=BEARER)
        check("POST /api/v1/evaluate -> 200", r.status_code == 200, r.text[:200])

        # ------------------------------------------------------ idempotency
        print("\n== idempotency ==")
        hdr = {**AUTH, "X-Request-Id": f"req-{int(time.time() * 1000)}"}
        r1 = client.post("/evaluate", files=files, headers=hdr)
        r2 = client.post("/evaluate", files=files, headers=hdr)
        check("first request -> 200", r1.status_code == 200, r1.text[:200])
        check("repeat request -> same id",
              r1.json()["id"] == r2.json()["id"],
              f"{r1.json().get('id')} vs {r2.json().get('id')}")
        check("repeat marked idempotent", r2.json().get("idempotent") is True, r2.text[:200])

        # ------------------------------------------------------- validation
        print("\n== validation ==")
        bad = {"source_audio": ("a.txt", b"hello", "text/plain"),
               "target_audio": ("b.wav", wav(), "audio/wav")}
        r = client.post("/evaluate", files=bad, headers=AUTH)
        check("wrong extension -> 415", r.status_code == 415, r.text[:200])
        check("415 code UNSUPPORTED_AUDIO",
              r.json().get("error", {}).get("code") == "UNSUPPORTED_AUDIO", r.text[:200])

        corrupt = {"source_audio": ("broken.wav", testkit.corrupt_wav_bytes(), "audio/wav"),
                   "target_audio": ("ok.wav", wav(), "audio/wav")}
        r = client.post("/evaluate", files=corrupt, headers=AUTH)
        check("corrupt audio -> 422", r.status_code == 422, r.text[:250])
        check("422 code VALIDATION_ERROR",
              r.json().get("error", {}).get("code") == "VALIDATION_ERROR", r.text[:250])

        r = client.post("/evaluate", files=files, headers=AUTH,
                        params={"min_similarity": 5})
        check("out-of-range parameter -> 422", r.status_code == 422, r.text[:200])

        big = {"source_audio": ("big.wav", b"\x00" * (201 * 1024 * 1024), "audio/wav"),
               "target_audio": ("b.wav", wav(), "audio/wav")}
        r = client.post("/evaluate", files=big, headers=AUTH)
        check("oversized upload -> 413", r.status_code == 413, r.text[:200])
        check("413 code PAYLOAD_TOO_LARGE",
              r.json().get("error", {}).get("code") == "PAYLOAD_TOO_LARGE", r.text[:200])

        # ------------------------------------------------------- resampling
        print("\n== audio normalisation ==")
        FAKES.seen_audio.clear()
        eight_k = {"source_audio": ("EN_8k.wav", wav(seconds=1, rate=8000), "audio/wav"),
                   "target_audio": ("ES_8k.wav", wav(seconds=1, rate=8000), "audio/wav")}
        r = client.post("/evaluate", files=eight_k, headers=AUTH)
        check("8 kHz upload accepted", r.status_code == 200, r.text[:250])
        if FAKES.seen_audio:
            probe = FAKES.seen_audio[-1]
            check("8 kHz source resampled to 16 kHz",
                  probe.get("source_sample_rate") == 16000, str(probe))
            check("mono preserved", probe.get("source_channels") == 1, str(probe))
        else:
            check("engine received the audio", False, "no probe recorded")

        # ------------------------------------------------------------- jobs
        print("\n== jobs ==")
        r = client.post("/jobs", files=files, headers=AUTH)
        check("POST /jobs -> 202", r.status_code == 202, r.text[:300])
        job = r.json()
        check("job has job_id", "job_id" in job, r.text[:200])
        check("job status queued", job.get("status") == "queued", r.text[:200])
        job_id = job["job_id"]

        final = None
        for _ in range(100):
            time.sleep(0.05)
            jr = client.get(f"/jobs/{job_id}", headers=AUTH)
            if jr.status_code == 200 and jr.json()["status"] in ("done", "failed"):
                final = jr.json()
                break
        check("job reached done", final is not None and final["status"] == "done",
              json.dumps(final)[:300] if final else "timeout")
        if final:
            check("job state alias is 'completed'", final.get("state") == "completed",
                  json.dumps(final)[:200])
            check("job carries the result",
                  (final.get("result") or {}).get("summary", {}).get("chunks") == 2)
            check("job result carries the SER score",
                  (final.get("result") or {}).get("ser_match_score") == 0.92)

        r = client.get("/jobs/nope", headers=AUTH)
        check("unknown job -> 404", r.status_code == 404, r.text[:200])

        r = client.get("/jobs", headers=AUTH)
        check("GET /jobs -> items + counts", "items" in r.json() and "counts" in r.json(),
              r.text[:200])
        check("job counts include the spec aliases",
              "completed" in r.json()["counts"] and "pending" in r.json()["counts"],
              json.dumps(r.json().get("counts")))

        jhdr = {**AUTH, "X-Request-Id": f"job-req-{int(time.time() * 1000)}"}
        j1 = client.post("/jobs", files=files, headers=jhdr)
        j2 = client.post("/jobs", files=files, headers=jhdr)
        check("repeat job -> same job_id",
              j1.json()["job_id"] == j2.json()["job_id"],
              f"{j1.json().get('job_id')} vs {j2.json().get('job_id')}")
        check("repeat job marked idempotent", j2.json().get("idempotent") is True, j2.text[:200])

        # ---------------------------------------------- poison-pill isolation
        print("\n== poison-pill isolation ==")
        # Driven directly so the two jobs have paths we control - the API
        # renames uploads to upload_<uuid><ext>, so a filename marker would
        # never reach the engine.
        import tempfile

        def _tmp_wav():
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.write(wav())
            handle.close()
            return handle.name

        poison_src, poison_tgt = _tmp_wav(), _tmp_wav()
        ok_src, ok_tgt = _tmp_wav(), _tmp_wav()
        FAKES.fail_keys = [poison_src]

        def _job(path_src, path_tgt, name):
            job_id, _created = EVAL_STORE.create_job(
                source_name=name, target_name=name + ".target")
            return {"job_id": job_id, "source_path": path_src,
                    "target_path": path_tgt, "source_name": name,
                    "target_name": name + ".target", "request_id": None,
                    "min_similarity": 0.5, "max_merge": 2}

        job_poison = _job(poison_src, poison_tgt, "POISON.wav")
        job_ok = _job(ok_src, ok_tgt, "OK.wav")

        API._process_batch([job_poison, job_ok])

        rec_poison = EVAL_STORE.get_job(job_poison["job_id"])
        rec_ok = EVAL_STORE.get_job(job_ok["job_id"])
        check("poison job failed", rec_poison["status"] == "failed",
              json.dumps(rec_poison)[:250])
        check("its error carries a code",
              (rec_poison.get("error") or {}).get("code") == "GPU_OUT_OF_MEMORY",
              json.dumps(rec_poison.get("error"))[:250])
        check("its error is retryable",
              (rec_poison.get("error") or {}).get("retryable") is True)
        check("the healthy job in the same batch still succeeded",
              rec_ok["status"] == "done", json.dumps(rec_ok)[:250])
        FAKES.fail_keys = []

        # ------------------------------------------------------- the store
        print("\n== stored evaluations ==")
        r = client.get("/evaluations", headers=AUTH)
        check("GET /evaluations -> items/total/limit/offset",
              all(k in r.json() for k in ("items", "total", "limit", "offset")), r.text[:200])
        check("total counts every evaluation", r.json()["total"] >= 3, r.json()["total"])

        r = client.get("/evaluations", headers=AUTH, params={"limit": 2})
        check("limit honoured", len(r.json()["items"]) <= 2)
        r = client.get("/evaluations", headers=AUTH, params={"limit": 1, "offset": 1})
        check("offset honoured", len(r.json()["items"]) == 1, r.text[:200])
        r = client.get("/evaluations", headers=AUTH, params={"source_name": "EN_001.wav"})
        check("source_name filter works",
              all(i["source_name"] == "EN_001.wav" for i in r.json()["items"]), r.text[:200])
        r = client.get("/evaluations", headers=AUTH, params={"order": "asc"})
        check("order=asc accepted", r.status_code == 200, r.text[:200])
        r = client.get("/evaluations", headers=AUTH, params={"since": 0})
        check("since filter accepted", r.status_code == 200, r.text[:200])

        r = client.get(f"/evaluations/{eval_id}", headers=AUTH)
        check("GET /evaluations/{id} -> full record",
              r.status_code == 200 and "result" in r.json(), r.text[:200])
        r = client.get("/evaluations/latest", headers=AUTH)
        check("GET /evaluations/latest -> 200", r.status_code == 200, r.text[:200])
        r = client.get("/evaluations/does-not-exist", headers=AUTH)
        check("unknown evaluation -> 404", r.status_code == 404, r.text[:200])

        r = client.get("/evaluations/export", headers=AUTH, params={"format": "jsonl"})
        check("export jsonl -> 200 ndjson",
              r.status_code == 200 and "x-ndjson" in r.headers.get("content-type", ""),
              r.text[:200])
        check("export jsonl has rows", len(r.text.strip().splitlines()) >= 1)
        r = client.get("/evaluations/export", headers=AUTH, params={"format": "csv"})
        check("export csv -> 200", r.status_code == 200 and "text/csv" in r.headers.get("content-type", ""))
        check("export csv header", r.text.splitlines()[0].startswith("id,created_at"),
              r.text[:120])

        r = client.delete(f"/evaluations/{eval_id}", headers=AUTH)
        check("DELETE evaluation -> 200", r.status_code == 200, r.text[:200])
        r = client.get(f"/evaluations/{eval_id}", headers=AUTH)
        check("deleted evaluation -> 404", r.status_code == 404, r.text[:200])
        r = client.delete(f"/evaluations/{eval_id}", headers=AUTH)
        check("delete again -> 404", r.status_code == 404, r.text[:200])

        # ------------------------------------------------------ prune/retention
        print("\n== retention ==")
        r = client.post("/evaluations/prune", headers=AUTH, params={"before": time.time() + 60})
        check("prune -> 200", r.status_code == 200, r.text[:200])
        check("prune removed everything older than the cutoff",
              r.json()["deleted"] >= 1, r.text[:200])

        # ---------------------------------------------------------- metrics
        print("\n== metrics ==")
        r = client.get("/metrics", headers=AUTH)
        check("GET /metrics -> 200", r.status_code == 200, r.text[:200])
        m = r.json()
        check("metrics count evaluations", m.get("evaluations_total", 0) >= 1, json.dumps(m)[:250])
        check("metrics expose a rolling quality window", "quality_window" in m, json.dumps(m)[:250])
        qw = m.get("quality_window") or {}
        check("rolling window has a COMET mean", qw.get("avg_comet_kiwi") is not None, json.dumps(qw)[:250])
        check("rolling window has latency percentiles",
              qw.get("latency_p50_s") is not None, json.dumps(qw)[:250])
        check("metrics include the SER mean", "avg_ser_match_score" in qw, json.dumps(qw)[:250])

        r = client.get("/metrics", headers=AUTH, params={"format": "prometheus"})
        check("GET /metrics?format=prometheus -> 200",
              r.status_code == 200 and "text/plain" in r.headers.get("content-type", ""),
              r.headers.get("content-type", ""))
        for metric in ("compnova_evaluations_total", "compnova_queue_depth",
                       "compnova_uptime_seconds", "compnova_rolling_comet_kiwi"):
            check(f"prometheus exports {metric}", metric in r.text, r.text[:200])

        r = client.get("/metrics/prometheus", headers=AUTH)
        check("GET /metrics/prometheus -> 200", r.status_code == 200, r.text[:120])

        # --------------------------------------------------------- translate
        print("\n== translate ==")
        r = client.post("/translate", files={"source_audio": ("EN_1.wav", wav(), "audio/wav")},
                        data={"target_lang": "spa"}, headers=AUTH)
        check("POST /translate -> 200", r.status_code == 200, r.text[:250])
        check("translate returns the text", r.json().get("text") == "translated to spa",
              r.text[:200])
        r = client.post("/translate", files={"source_audio": ("EN_1.wav", wav(), "audio/wav")},
                        data={"target_lang": "spa", "mode": "nonsense"}, headers=AUTH)
        check("bad translation mode -> 422", r.status_code == 422, r.text[:200])

        # --------------------------------------------------- stale job recovery
        print("\n== startup recovery ==")
        stale_id, _ = EVAL_STORE.create_job(source_name="stale.wav", target_name="stale.wav")
        EVAL_STORE.update_job(stale_id, status="running")
        n = EVAL_STORE.fail_stale_jobs_on_startup()
        rec = EVAL_STORE.get_job(stale_id)
        check("fail_stale_jobs_on_startup marks in-flight jobs failed",
              rec["status"] == "failed", json.dumps(rec)[:200])
        check("the stale job explains why",
              (rec.get("error") or {}).get("code") == "SERVICE_RESTARTED",
              json.dumps(rec.get("error"))[:200])
        check("the stale job is retryable", (rec.get("error") or {}).get("retryable") is True)

        # --------------------------------------------------- audio quality
        print("\n== audio quality ==")
        r = client.post("/audio/quality",
                        files={"audio": ("EN_1.wav", wav(), "audio/wav")},
                        headers=AUTH)
        check("POST /audio/quality -> 200", r.status_code == 200, r.text[:250])
        body = r.json() if r.status_code == 200 else {}
        check("the quality report has a source section",
              "source" in body, r.text[:200])
        check("the quality report has flags", isinstance(body.get("flags"), list),
              r.text[:200])
        check("the quality report measures the duration",
              isinstance(body.get("source", {}).get("duration_s"), (int, float)),
              r.text[:200])
        r = client.post("/audio/quality",
                        files={"audio": ("EN_1.wav", wav(), "audio/wav"),
                               "target_audio": ("ES_1.wav", wav(), "audio/wav")},
                        headers=AUTH)
        check("POST /audio/quality with a target -> 200", r.status_code == 200,
              r.text[:250])
        check("the pair report has both sides",
              "source" in r.json() and "target" in r.json(), r.text[:200])
        r = client.post("/audio/quality",
                        files={"audio": ("EN_1.wav", b"not audio", "audio/wav")},
                        headers=AUTH)
        check("a corrupt file is rejected, not scored", r.status_code >= 400,
              r.text[:200])

        # ----------------------------------------------------------- openapi
        print("\n== openapi ==")
        r = client.get("/openapi.json")
        check("openapi served", r.status_code == 200)
        paths = r.json().get("paths", {})
        for p in ("/health", "/health/ready", "/evaluate", "/jobs", "/evaluations",
                  "/evaluations/{eval_id}", "/evaluations/export", "/metrics",
                  "/metrics/prometheus", "/translate", "/transcribe",
                  "/audio/quality"):
            check(f"openapi documents {p}", p in paths, str(list(paths)))
        check("openapi documents the /api/v1 prefix", "/api/v1/evaluate" in paths,
              str(list(paths)))

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

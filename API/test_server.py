"""
Boots the evaluation service as a REAL Uvicorn server and drives it over HTTP.

The in-process suite (test_api.py) uses FastAPI's TestClient, which bypasses
the socket layer. This one starts an actual server process, so it also proves
that the module imports cleanly, that the lifespan handler runs, that the
middleware is wired up, and that the app binds a port.

Still no GPU and no model weights: the model loaders are stubbed.

    python3 API/test_server.py                 # run the checks
    python3 API/test_server.py --serve --port 8767   # just run the server
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(port: int) -> None:
    """Child-process entry point: stub the models, then run Uvicorn."""
    import testkit

    testkit.install_everything("/tmp/compnova_server_test/evaluations.db")
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=port, log_level="warning")


def main() -> int:
    if "--serve" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8767
        serve(port)
        return 0

    import requests

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--serve", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    try:
        print("== booting the server ==")
        ready = False
        for _ in range(120):
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout else ""
                check("the server process stays alive", False, output[-800:])
                return 1
            try:
                if requests.get(f"{base}/health", timeout=1).ok:
                    ready = True
                    break
            except requests.RequestException:
                time.sleep(0.5)
        check("the server boots and answers /health", ready,
              "no answer within 60s")
        if not ready:
            return 1

        auth = {"X-API-Key": "test-key-1"}
        bearer = {"Authorization": "Bearer test-key-2"}

        print("\n== live HTTP surface ==")
        r = requests.get(f"{base}/health", timeout=5)
        check("GET /health -> 200 ok", r.status_code == 200 and r.json()["status"] == "ok",
              r.text)
        check("X-API-Version header is sent", r.headers.get("X-API-Version") == "2.1.0",
              str(dict(r.headers)))
        r = requests.get(f"{base}/api/v1/health", timeout=5)
        check("GET /api/v1/health -> 200 ok", r.status_code == 200, r.text)
        r = requests.get(f"{base}/health/ready", timeout=5)
        check("GET /health/ready -> 200 ready",
              r.status_code == 200 and r.json()["status"] == "ready", r.text[:250])
        check("readiness reports the models",
              "whisper" in r.json().get("models", {}), r.text[:250])

        r = requests.get(f"{base}/evaluations", timeout=5)
        check("no key -> 401", r.status_code == 401, r.text[:200])
        check("401 body has error.code",
              r.json().get("error", {}).get("code") == "UNAUTHORIZED", r.text[:200])
        r = requests.get(f"{base}/evaluations", headers=bearer, timeout=5)
        check("Bearer token -> 200", r.status_code == 200, r.text[:200])

        import testkit

        def _files():
            return {
                "source_audio": ("EN_001.wav", testkit.wav_bytes(), "audio/wav"),
                "target_audio": ("ES_001.wav", testkit.wav_bytes(), "audio/wav"),
            }

        r = requests.post(f"{base}/evaluate", files=_files(), headers=auth, timeout=60)
        check("POST /evaluate -> 200", r.status_code == 200, r.text[:300])
        body = r.json()
        check("response carries an id and a summary",
              "id" in body and body.get("summary", {}).get("chunks") == 2, r.text[:250])
        check("response carries the SER score", body.get("ser_match_score") == 0.92,
              r.text[:250])

        r = requests.post(f"{base}/jobs", files=_files(), headers=auth, timeout=30)
        check("POST /jobs -> 202", r.status_code == 202, r.text[:250])
        job_id = r.json().get("job_id")
        final = None
        for _ in range(60):
            time.sleep(0.5)
            jr = requests.get(f"{base}/jobs/{job_id}", headers=auth, timeout=10)
            if jr.json().get("status") in ("done", "failed"):
                final = jr.json()
                break
        check("the job completes over HTTP",
              final is not None and final.get("status") == "done",
              str(final)[:250])

        r = requests.get(f"{base}/metrics", headers=auth, timeout=10)
        check("GET /metrics -> 200", r.status_code == 200, r.text[:200])
        r = requests.get(f"{base}/metrics/prometheus", headers=auth, timeout=10)
        check("GET /metrics/prometheus -> 200 text",
              r.status_code == 200 and "compnova_evaluations_total" in r.text,
              r.text[:200])

        r = requests.get(f"{base}/openapi.json", timeout=10)
        check("GET /openapi.json -> 200", r.status_code == 200, r.text[:120])
        check("openapi documents /api/v1/evaluate",
              "/api/v1/evaluate" in r.json().get("paths", {}), "")

        print("\n== shutdown ==")
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
        check("the server stops cleanly", child.poll() is not None, "")

    finally:
        if child.poll() is None:
            child.kill()

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

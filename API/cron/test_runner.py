"""
End-to-end tests for the batch runner (run_batch.py) against the mock API.

Starts API/cron/mock_api.py as a real server, then drives the runner over a
real inbox directory and checks what ends up in done/, failed/ and results/.

No GPU, no models, no network.

    python3 API/cron/test_runner.py
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_batch.py")
MOCK = os.path.join(HERE, "mock_api.py")

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


def start_mock(port: int, slow: float = 1.0, api_keys: str = "") -> subprocess.Popen:
    env = dict(os.environ)
    if api_keys:
        env["MOCK_API_KEYS"] = api_keys
    else:
        env.pop("MOCK_API_KEYS", None)
    return subprocess.Popen(
        [sys.executable, MOCK, str(port), str(slow)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )


def wait_for(port: int, timeout: float = 20.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    import struct

    frames = int(rate * seconds)
    data = b"\x00\x00" * frames
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def run_runner(inbox, port, *extra, api_key=None):
    cmd = [sys.executable, RUNNER, "--inbox", str(inbox),
           "--api-url", f"http://127.0.0.1:{port}", "--poll-interval", "0.5"]
    if api_key:
        cmd += ["--api-key", api_key]
    cmd += list(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return proc


def fresh_pipeline() -> str:
    root = tempfile.mkdtemp(prefix="cn-pipeline-")
    for name in ("inbox", "processing", "done", "failed", "results"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    return root


def main() -> int:
    port = free_port()
    mock = start_mock(port, slow=1.0)
    try:
        if not wait_for(port):
            print("the mock API never came up")
            return 1

        # ------------------------------------------------- pairing & dry run
        print("\n== pairing and --dry-run ==")
        root = fresh_pipeline()
        inbox = os.path.join(root, "inbox")
        for name in ("EN_001.wav", "ES_001.wav", "EN_002.wav", "ES_002.wav",
                     "EN_003.wav"):
            with open(os.path.join(inbox, name), "wb") as fh:
                fh.write(wav())
        proc = run_runner(inbox, port, "--dry-run")
        out = proc.stdout + proc.stderr
        check("--dry-run pairs EN with ES", "would evaluate EN_001.wav + ES_001.wav" in out,
              out[-400:])
        check("--dry-run sends nothing", "-> id=" not in out, out[-400:])
        check("--dry-run reports the unpaired file",
              "skip (no counterpart) EN_003.wav" in out, out[-400:])
        proc_src = run_runner(inbox, port, "--dry-run", "--items", "sources")
        check("--dry-run --items sources works", proc_src.returncode == 0,
              f"rc={proc_src.returncode} {proc_src.stdout[-300:]} {proc_src.stderr[-300:]}")

        # ------------------------------------------------------- job mode
        print("\n== job mode ==")
        proc = run_runner(inbox, port, "--mode", "job")
        out = proc.stdout + proc.stderr
        check("the runner succeeds", proc.returncode == 0, out[-600:])
        check("both pairs are evaluated",
              "001 -> id=" in out and "002 -> id=" in out, out[-600:])
        done = os.listdir(os.path.join(root, "done"))
        check("both files of a pair land in done/",
              "EN_001.wav" in done and "ES_001.wav" in done, str(done))
        check("the unpaired file stays in the inbox",
              "EN_003.wav" in os.listdir(inbox), str(os.listdir(inbox)))
        results = os.listdir(os.path.join(root, "results"))
        ledgers = [f for f in results if f.endswith(".json")]
        check("a ledger entry per evaluated pair", len(ledgers) == 2, str(results))
        if ledgers:
            record = json.load(open(os.path.join(root, "results", ledgers[0])))
            check("the ledger carries the summary",
                  record.get("summary", {}).get("chunks") == 3, json.dumps(record)[:250])
            check("the ledger carries the SER score",
                  record.get("ser_match_score") == 0.88, json.dumps(record)[:250])

        print("\n== re-running skips completed work ==")
        # Put the pair back: the .done marker keyed on the pair name means the
        # runner skips it instead of evaluating it a second time.
        for name in ("EN_001.wav", "ES_001.wav"):
            with open(os.path.join(inbox, name), "wb") as fh:
                fh.write(wav())
        proc = run_runner(inbox, port)
        out = proc.stdout + proc.stderr
        check("already-evaluated files are skipped", "already evaluated" in out, out[-400:])
        check("the skipped pair is not evaluated again",
              "001 -> id=" not in out, out[-400:])

        # ---------------------------------------------- source-only mode
        print("\n== source-only mode ==")
        root2 = fresh_pipeline()
        inbox2 = os.path.join(root2, "inbox")
        with open(os.path.join(inbox2, "EN_010.wav"), "wb") as fh:
            fh.write(wav())
        proc = run_runner(inbox2, port, "--items", "sources", "--target-lang", "spa")
        out = proc.stdout + proc.stderr
        check("a lone source file is submitted for translation",
              "010 -> id=" in out, out[-500:])
        check("the source file lands in done/",
              "EN_010.wav" in os.listdir(os.path.join(root2, "done")),
              str(os.listdir(os.path.join(root2, "done"))))

        # ----------------------------------------- corrupt file dead-letter
        print("\n== corrupt audio is dead-lettered ==")
        root3 = fresh_pipeline()
        inbox3 = os.path.join(root3, "inbox")
        with open(os.path.join(inbox3, "EN_020.wav"), "wb") as fh:
            fh.write(wav())
        with open(os.path.join(inbox3, "ES_020.wav"), "wb") as fh:
            fh.write(b"RIFF\x10\x00\x00\x00WAVEjunkjunkjunkjunk")
        proc = run_runner(inbox3, port)
        out = proc.stdout + proc.stderr
        check("the corrupt file is rejected before upload",
              "pre-flight validation" in out, out[-500:])
        check("nothing was sent to the API for the corrupt pair",
              "020 -> id=" not in out, out[-500:])
        failed = os.listdir(os.path.join(root3, "failed"))
        check("the corrupt pair lands in failed/",
              "ES_020.wav" in failed, str(failed))
        check("a reason is recorded",
              any(f.endswith(".error.txt") for f in failed), str(failed))

        # -------------------------------------------------- stale recovery
        print("\n== stale processing/ recovery ==")
        root4 = fresh_pipeline()
        inbox4 = os.path.join(root4, "inbox")
        stale = os.path.join(root4, "processing", "EN_030.wav")
        with open(stale, "wb") as fh:
            fh.write(wav())
        old = time.time() - (3600 * 100)
        os.utime(stale, (old, old))
        # Its partner is already in the inbox, so the recovered file forms a pair.
        with open(os.path.join(inbox4, "ES_030.wav"), "wb") as fh:
            fh.write(wav())
        proc = run_runner(inbox4, port)
        out = proc.stdout + proc.stderr
        check("a stranded file is recovered", "recovered 1 stale file" in out, out[-400:])
        check("the recovered file is evaluated again", "030 -> id=" in out, out[-400:])

        # -------------------------------------------------------- auth
        print("\n== an auth failure returns the files to the inbox ==")
        auth_port = free_port()
        auth_mock = start_mock(auth_port, slow=0.2, api_keys="secret-key")
        try:
            if wait_for(auth_port):
                root5 = fresh_pipeline()
                inbox5 = os.path.join(root5, "inbox")
                for name in ("EN_040.wav", "ES_040.wav"):
                    with open(os.path.join(inbox5, name), "wb") as fh:
                        fh.write(wav())
                proc = run_runner(inbox5, auth_port, "--max-attempts", "1")
                out = proc.stdout + proc.stderr
                check("the runner reports the 401", "UNAUTHORIZED" in out, out[-400:])
                check("the files are back in the inbox, not in failed/",
                      "EN_040.wav" in os.listdir(inbox5),
                      str(os.listdir(inbox5)))
                check("nothing was written to failed/",
                      not [f for f in os.listdir(os.path.join(root5, "failed"))
                           if f.endswith(".wav")],
                      str(os.listdir(os.path.join(root5, "failed"))))

                print("\n== the right key works ==")
                proc = run_runner(inbox5, auth_port, "--max-attempts", "1",
                                  api_key="secret-key")
                out = proc.stdout + proc.stderr
                check("the runner succeeds with the right key",
                      "040 -> id=" in out, out[-400:])
        finally:
            auth_mock.terminate()
            try:
                auth_mock.wait(timeout=10)
            except subprocess.TimeoutExpired:
                auth_mock.kill()

    finally:
        mock.terminate()
        try:
            mock.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mock.kill()

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
CompNova evaluation batch runner — reference implementation for the
"audio lands in a folder, a cron job evaluates it" pipeline.

The service evaluates a pair of audio files (source + translated), or takes
a source file on its own and translates it first. This script:

  1. recovers any stale files stranded in processing/ from an interrupted run,
  2. claims files out of the inbox atomically (so two cron runs never collide),
  3. validates audio integrity locally before uploading (routing corrupted
     files straight to failed/ as a dead-letter queue),
  4. pairs source and target files by their naming convention (or treats
     every file as a source to be translated when --items sources or
     --translate-unpaired is set),
  5. submits each item to the API and waits for the result,
  6. reconciles against the API on a timeout instead of blindly retrying,
  7. writes a durable local ledger to results/<eval_id>.json,
  8. files the inputs into done/ or failed/ with a reason, and optionally
     purges archived files older than --retention-days.

Only the standard library plus `requests` is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("run_batch.py needs `requests` (pip install requests)")

LOG = logging.getLogger("compnova.eval.batch")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"}


# --------------------------------------------------------------------------- #
# local pre-flight audio validation
# --------------------------------------------------------------------------- #

def validate_local_audio(path: Path) -> tuple[bool, str]:
    """Verify file integrity (non-empty, readable audio container/header, non-zero frames)."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False, f"{path.name} is missing or empty (0 bytes)"
    except OSError as exc:
        return False, f"cannot stat {path.name}: {exc}"

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.getnframes()
                if sr <= 0 or ch <= 0 or frames <= 0:
                    return False, f"{path.name} has invalid WAV parameters (sr={sr}, channels={ch}, frames={frames})"
                return True, f"WAV {sr}Hz {ch}ch ({frames / sr:.1f}s)"
        except Exception as wav_exc:
            # Could be a non-PCM WAV or compressed container; check ffprobe if available
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                    check=True, capture_output=True, text=True, timeout=15,
                )
                if float(out.stdout.strip() or 0) > 0:
                    return True, "validated via ffprobe"
            except Exception:
                pass
            return False, f"corrupted WAV file {path.name}: {wav_exc}"

    return True, "ok"


# --------------------------------------------------------------------------- #
# pairing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Pair:
    source: Path
    target: Path | None
    key: str

    @property
    def is_source_only(self) -> bool:
        return self.target is None


def split_prefixed(name: str, src_lang: str, tgt_lang: str):
    """`EN_001_#1.wav` with src=EN, tgt=ES  ->  ("001_#1", "EN")."""
    stem = Path(name).stem
    low = stem.lower()
    for lang in (src_lang, tgt_lang):
        ll = lang.lower()
        if low.startswith(ll):
            rest = stem[len(lang):]
            if rest[:1] in ("_", "-", ".", " "):
                return rest[1:], lang.upper()
    return None


def split_dotted(name: str, src_lang: str, tgt_lang: str):
    """`clip_001.EN.wav` with src=EN, tgt=ES  ->  ("clip_001", "EN")."""
    parts = Path(name).stem.split(".")
    if len(parts) < 2:
        return None
    tail = parts[-1].upper()
    if tail not in (src_lang.upper(), tgt_lang.upper()):
        return None
    return ".".join(parts[:-1]), tail


def find_pairs(inbox: Path, src_lang: str, tgt_lang: str, mode: str,
               translate_unpaired: bool = False):
    """Return (complete pairs, files that have no counterpart yet)."""
    splitter = split_prefixed if mode == "prefix" else split_dotted
    by_key: dict[str, dict[str, Path]] = {}
    strays: list[Path] = []

    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.name.startswith("."):
            continue
        parsed = splitter(path.name, src_lang, tgt_lang)
        if parsed is None:
            strays.append(path)
            continue
        key, lang = parsed
        by_key.setdefault(key, {})[lang] = path

    pairs: list[Pair] = []
    for key, langs in sorted(by_key.items()):
        src = langs.get(src_lang.upper())
        tgt = langs.get(tgt_lang.upper())
        if src is not None and tgt is not None:
            pairs.append(Pair(src, tgt, key))
        elif src is not None and translate_unpaired:
            pairs.append(Pair(src, None, key))
        else:
            strays.extend(langs.values())
    return pairs, strays


def find_sources(inbox: Path, src_lang: str, mode: str):
    """Every audio file in the inbox is a source to be translated."""
    sources: list[Pair] = []
    strays: list[Path] = []
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.name.startswith("."):
            continue
        if mode == "prefix":
            parsed = split_prefixed(path.name, src_lang, src_lang)
        else:
            parsed = split_dotted(path.name, src_lang, src_lang)
        key = parsed[0] if parsed else path.stem
        sources.append(Pair(path, None, key))
    return sources, strays


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

class ApiError(RuntimeError):
    """The API returned an error response."""

    def __init__(self, status: int, detail: str, retryable: bool, code: str = ""):
        super().__init__(f"HTTP {status} {code}: {detail}")
        self.status = status
        self.detail = detail
        self.retryable = retryable
        self.code = code


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None, connect_timeout: float,
                 read_timeout: float):
        base = base_url.rstrip("/")
        # Normalize if caller passed .../api/v1/evaluate or .../evaluate
        for suffix in ("/api/v1/evaluate", "/evaluate", "/api/v1"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        self.base_url = base
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.session = requests.Session()

    def _headers(self, request_id: str | None = None) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        if request_id:
            headers["X-Request-Id"] = request_id
        return headers

    def health(self) -> bool:
        try:
            r = self.session.get(
                f"{self.base_url}/health",
                timeout=(self.connect_timeout, self.connect_timeout),
            )
            return r.ok
        except requests.RequestException:
            return False

    def ready(self) -> tuple[bool, dict]:
        """Readiness: are the models loaded? Returns (ready, report)."""
        try:
            r = self.session.get(
                f"{self.base_url}/health/ready",
                timeout=(self.connect_timeout, self.read_timeout),
            )
            report = r.json() if r.headers.get("content-type", "").startswith(
                "application/json") else {}
            return r.status_code == 200, report
        except (requests.RequestException, ValueError):
            return False, {}

    @staticmethod
    def _error_from(response) -> ApiError:
        code, detail, retryable = "", response.text[:500], response.status_code >= 500
        try:
            body = response.json()
            err = body.get("error", {})
            code = err.get("code", "")
            detail = err.get("message", detail)
            retryable = bool(err.get("retryable", retryable))
        except ValueError:
            pass
        return ApiError(response.status_code, str(detail), retryable, code)

    def submit_job(self, pair: Pair, target_lang: str, min_similarity: float,
                   max_merge: int, request_id: str) -> str:
        """POST /jobs -> job_id."""
        params = {"min_similarity": min_similarity, "max_merge": max_merge}
        files = {"source_audio": (pair.source.name, pair.source.open("rb"),
                                  "application/octet-stream")}
        try:
            if pair.target is not None:
                files["target_audio"] = (pair.target.name, pair.target.open("rb"),
                                         "application/octet-stream")
            data = {"target_lang": target_lang}
            r = self.session.post(
                f"{self.base_url}/jobs",
                params=params,
                files=files,
                data=data,
                headers=self._headers(request_id),
                timeout=(self.connect_timeout, self.read_timeout),
            )
        finally:
            for handle in files.values():
                handle[1].close()
        if not r.ok:
            raise self._error_from(r)
        return r.json()["job_id"]

    def get_job(self, job_id: str) -> dict:
        r = self.session.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=(self.connect_timeout, self.read_timeout),
            headers=self._headers(),
        )
        if r.status_code == 404:
            raise ApiError(404, f"job {job_id} not found", False, "NOT_FOUND")
        if not r.ok:
            raise self._error_from(r)
        return r.json()

    def evaluate_sync(self, pair: Pair, min_similarity: float, max_merge: int,
                      request_id: str) -> dict:
        params = {"min_similarity": min_similarity, "max_merge": max_merge}
        files = {
            "source_audio": (pair.source.name, pair.source.open("rb"),
                             "application/octet-stream"),
        }
        try:
            if pair.target is not None:
                files["target_audio"] = (pair.target.name, pair.target.open("rb"),
                                         "application/octet-stream")
            r = self.session.post(
                f"{self.base_url}/evaluate",
                params=params,
                files=files,
                headers=self._headers(request_id),
                timeout=(self.connect_timeout, self.read_timeout),
            )
        finally:
            for handle in files.values():
                handle[1].close()
        if not r.ok:
            raise self._error_from(r)
        return r.json()

    def find_evaluation_by_names(self, source_name: str, target_name: str | None,
                                 not_before: float) -> dict | None:
        """Reconciliation: did our evaluation finish despite a client timeout?"""
        try:
            r = self.session.get(
                f"{self.base_url}/evaluations",
                params={"limit": 50, "source_name": source_name, "since": not_before},
                timeout=(self.connect_timeout, self.read_timeout),
                headers=self._headers(),
            )
            r.raise_for_status()
            body = r.json()
        except requests.RequestException:
            return None
        items = body.get("items", body) if isinstance(body, dict) else body
        for item in items:
            if item.get("source_name") != source_name:
                continue
            if target_name and item.get("target_name") != target_name:
                continue
            if float(item.get("created_at", 0)) < not_before:
                continue
            try:
                rr = self.session.get(
                    f"{self.base_url}/evaluations/{item['id']}",
                    timeout=(self.connect_timeout, self.read_timeout),
                    headers=self._headers(),
                )
                rr.raise_for_status()
                return rr.json()
            except requests.RequestException:
                return None
        return None


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

def request_id_for(pair: Pair) -> str:
    """A stable idempotency key for one input pair."""
    parts = [pair.key]
    for path in (pair.source, pair.target):
        if path is None:
            continue
        try:
            stat = path.stat()
            parts += [path.name, str(stat.st_size), str(int(stat.st_mtime))]
        except OSError:
            parts += [path.name, "?", "?"]
    return "cn-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


class BatchRunner:
    def __init__(self, args, client: ApiClient):
        self.args = args
        self.client = client
        self.inbox = Path(args.inbox)
        self.processing = Path(args.processing or self.inbox.parent / "processing")
        self.done = Path(args.done or self.inbox.parent / "done")
        self.failed = Path(args.failed or self.inbox.parent / "failed")
        self.results = Path(args.results or self.inbox.parent / "results")
        for d in (self.processing, self.done, self.failed, self.results):
            d.mkdir(parents=True, exist_ok=True)

    # -- lifecycle / recovery / retention --------------------------------- #

    def recover_stale_processing(self) -> int:
        """Return files stranded in processing/ older than --stale-processing-min back to inbox/."""
        if self.args.stale_processing_min <= 0:
            return 0
        cutoff = time.time() - (self.args.stale_processing_min * 60)
        recovered = 0
        for p in sorted(self.processing.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    os.replace(p, self.inbox / p.name)
                    recovered += 1
            except OSError as exc:
                LOG.warning("could not recover %s from processing/: %s", p, exc)
        if recovered:
            LOG.info("recovered %d stale file(s) from processing/ back to inbox/", recovered)
        return recovered

    def purge_expired_archives(self) -> int:
        """Delete archived files in done/ and failed/ older than --retention-days."""
        if self.args.retention_days <= 0:
            return 0
        cutoff = time.time() - (self.args.retention_days * 86400)
        purged = 0
        for directory in (self.done, self.failed):
            for p in directory.iterdir():
                if not p.is_file():
                    continue
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                        purged += 1
                except OSError as exc:
                    LOG.warning("could not purge %s: %s", p, exc)
        if purged:
            LOG.info("retention sweep purged %d archived file(s) older than %d days",
                     purged, self.args.retention_days)
        return purged

    # -- file handling ---------------------------------------------------- #

    def claim(self, path: Path) -> Path | None:
        """Move a file into processing/ atomically. Returns the new path."""
        target = self.processing / path.name
        try:
            os.replace(path, target)
            return target
        except OSError as exc:
            LOG.warning("could not claim %s: %s", path, exc)
            return None

    def retire(self, pair: Pair, ok: bool) -> None:
        dest = self.done if ok else self.failed
        for p in (pair.source, pair.target):
            if p is not None and p.exists():
                try:
                    os.replace(p, dest / p.name)
                except OSError as exc:
                    LOG.warning("could not move %s to %s: %s", p, dest, exc)

    def already_done(self, pair: Pair) -> bool:
        return (self.results / f"{pair.key}.done").exists()

    # -- one item --------------------------------------------------------- #

    def process(self, pair: Pair) -> bool:
        if self.already_done(pair):
            LOG.info("skip %s — already evaluated", pair.key)
            self.retire(pair, ok=True)
            return True

        claimed_src = self.claim(pair.source)
        claimed_tgt = self.claim(pair.target) if pair.target else None
        if not claimed_src or (pair.target and not claimed_tgt):
            for orig, claimed in ((pair.source, claimed_src),
                                  (pair.target, claimed_tgt)):
                if claimed and claimed.exists():
                    os.replace(claimed, orig)
            return False
        pair = Pair(claimed_src, claimed_tgt, pair.key)

        # Pre-flight validation: reject corrupted audio into dead-letter queue
        # immediately without wasting retries or stalling the pipeline.
        for side_path in (pair.source, pair.target):
            if side_path is None:
                continue
            valid, reason = validate_local_audio(side_path)
            if not valid:
                LOG.error("%s failed pre-flight validation (%s)", pair.key, reason)
                self.fail(pair, f"pre-flight validation failed: {reason}")
                return False

        request_id = request_id_for(pair)

        backoff = 5
        for attempt in range(1, self.args.max_attempts + 1):
            started = time.time()
            try:
                result = self._submit_and_wait(pair, request_id, started)
            except ApiError as exc:
                LOG.error("%s failed (%s)", pair.key, exc)
                if exc.status in (401, 403):
                    self.return_to_inbox(pair)
                    return False
                if exc.retryable and attempt < self.args.max_attempts:
                    LOG.info("retrying in %ss", backoff)
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                self.fail(pair, str(exc))
                return False
            except requests.Timeout:
                LOG.warning("%s timed out after %ss", pair.key, self.args.read_timeout)
                record = self.client.find_evaluation_by_names(
                    pair.source.name,
                    pair.target.name if pair.target else None,
                    started,
                )
                if record:
                    LOG.info("%s did complete (id %s) — adopting it",
                             pair.key, record.get("id"))
                    self.finish(pair, record)
                    return True
                if attempt < self.args.max_attempts:
                    LOG.info("no matching record; retrying in %ss", backoff)
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                self.fail(pair, f"timeout after {self.args.read_timeout}s")
                return False
            except requests.RequestException as exc:
                LOG.error("%s transport error: %s", pair.key, exc)
                if attempt < self.args.max_attempts:
                    LOG.info("retrying in %ss", backoff)
                    time.sleep(backoff)
                    backoff *= 3
                    continue
                self.fail(pair, f"transport error: {exc}")
                return False

            self.finish(pair, result)
            return True
        return False

    def _submit_and_wait(self, pair: Pair, request_id: str, started: float) -> dict:
        if self.args.mode == "sync":
            return self.client.evaluate_sync(
                pair, self.args.min_similarity, self.args.max_merge, request_id
            )

        job_id = self.client.submit_job(
            pair, self.args.target_lang, self.args.min_similarity,
            self.args.max_merge, request_id,
        )
        LOG.info("%s -> job %s", pair.key, job_id)

        deadline = time.time() + self.args.job_timeout
        while time.time() < deadline:
            record = self.client.get_job(job_id)
            status = record.get("status")
            if status in ("done", "completed"):
                return record.get("result") or {}
            if status in ("failed", "cancelled"):
                error = record.get("error") or {}
                raise ApiError(500, error.get("message", status),
                               bool(error.get("retryable", False)),
                               error.get("code", "JOB_FAILED"))
            time.sleep(self.args.poll_interval)
        raise requests.Timeout(f"job {job_id} did not finish in {self.args.job_timeout}s")

    def finish(self, pair: Pair, result: dict) -> None:
        if not result:
            self.fail(pair, "the API returned an empty result")
            return
        payload = result.get("result") if isinstance(result.get("result"), dict) else result
        eval_id = payload.get("id") or result.get("id", "unknown")
        created_at = payload.get("created_at") or result.get("created_at")
        summary = payload.get("summary", {})
        ser_match = payload.get("ser_match_score", summary.get("ser_match_score"))
        LOG.info("%s -> id=%s comet=%s metricx=%s ser_match=%s chunks=%s unmatched=%s",
                 pair.key, eval_id, summary.get("comet_kiwi"),
                 summary.get("metricx_24"), ser_match,
                 summary.get("chunks"), summary.get("unmatched"))

        record = {
            "id": eval_id,
            "created_at": created_at,
            "source_file": pair.source.name,
            "target_file": pair.target.name if pair.target else None,
            "source_name": payload.get("source_name"),
            "target_name": payload.get("target_name"),
            "comet_score": payload.get("comet_score", summary.get("comet_kiwi")),
            "metricx_score": payload.get("metricx_score", summary.get("metricx_24")),
            "ser_match_score": ser_match,
            "source_emotion_top1": payload.get("source_emotion_top1", summary.get("source_emotion_top1")),
            "target_emotion_top1": payload.get("target_emotion_top1", summary.get("target_emotion_top1")),
            "summary": summary,
            "ser": payload.get("ser", {}),
            "chunks": payload.get("chunks", []),
            "transcripts": payload.get("transcripts", {}),
            "unmatched": payload.get("unmatched", {}),
            "timings": payload.get("timings", {}),
        }
        (self.results / f"{eval_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.results / f"{pair.key}.done").write_text(eval_id, encoding="utf-8")
        self.retire(pair, ok=True)

    def fail(self, pair: Pair, reason: str) -> None:
        (self.failed / f"{pair.key}.error.txt").write_text(
            f"{pair.key}\n{pair.source.name}\n"
            f"{pair.target.name if pair.target else '-'}\n\n{reason}\n",
            encoding="utf-8",
        )
        self.retire(pair, ok=False)

    def return_to_inbox(self, pair: Pair) -> None:
        """Put the files back: the problem was the request, not the audio."""
        for p in (pair.source, pair.target):
            if p is not None and p.exists():
                try:
                    os.replace(p, self.inbox / p.name)
                except OSError as exc:
                    LOG.warning("could not return %s to the inbox: %s", p, exc)

    # -- one pass --------------------------------------------------------- #

    def run_once(self) -> int:
        self.recover_stale_processing()
        self.purge_expired_archives()

        if self.args.items == "sources":
            pairs, strays = find_sources(self.inbox, self.args.src_lang,
                                         self.args.pair_mode)
        else:
            pairs, strays = find_pairs(
                self.inbox, self.args.src_lang, self.args.tgt_lang,
                self.args.pair_mode,
                translate_unpaired=self.args.translate_unpaired,
            )
        if strays:
            LOG.info("%d file(s) with no counterpart (skipped): %s",
                     len(strays), ", ".join(p.name for p in strays[:10]))
        if not pairs:
            LOG.info("nothing to do")
            return 0

        LOG.info("%d item(s) to evaluate", len(pairs))
        ok = failed = 0
        for pair in pairs:
            if self.process(pair):
                ok += 1
            else:
                failed += 1
        LOG.info("done — %d succeeded, %d failed", ok, failed)
        return failed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pick up audio from a folder and evaluate it via the CompNova API.")
    p.add_argument("--inbox", required=True, help="watched folder the producer drops files into")
    p.add_argument("--api-url", default=os.environ.get("EVAL_API_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("EVAL_API_KEY"),
                   help="sent as X-API-Key and Authorization: Bearer")
    p.add_argument("--mode", choices=["job", "sync"], default="job",
                   help="job: submit and poll (safe behind a proxy). sync: block on one request")
    p.add_argument("--items", choices=["pairs", "sources"], default="pairs",
                   help="pairs: the folder holds source+target pairs. "
                        "sources: the folder holds source audio only, the API translates it")
    p.add_argument("--translate-unpaired", action="store_true",
                   help="in pairs mode, submit unpaired source files for translation instead of waiting for their counterpart")
    p.add_argument("--src-lang", default="EN")
    p.add_argument("--tgt-lang", default="ES")
    p.add_argument("--target-lang", default="spa",
                   help="target language for translation when --items sources")
    p.add_argument("--pair-mode", choices=["prefix", "dotted"], default="prefix",
                   help="prefix: EN_001.wav   dotted: clip_001.EN.wav")
    p.add_argument("--processing", default=None, help="default: <inbox>/../processing")
    p.add_argument("--done", default=None, help="default: <inbox>/../done")
    p.add_argument("--failed", default=None, help="default: <inbox>/../failed")
    p.add_argument("--results", default=None, help="default: <inbox>/../results")
    p.add_argument("--min-similarity", type=float, default=0.5)
    p.add_argument("--max-merge", type=int, default=2)
    p.add_argument("--read-timeout", type=float, default=600.0,
                   help="per-request read timeout in seconds")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--job-timeout", type=float, default=3600.0,
                   help="how long to wait for a submitted job")
    p.add_argument("--poll-interval", type=float, default=5.0,
                   help="how often to poll a running job")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--stale-processing-min", type=float, default=120.0,
                   help="recover files stranded in processing/ older than this many minutes (0 = disable)")
    p.add_argument("--retention-days", type=int, default=int(os.environ.get("EVAL_RETENTION_DAYS", "0")),
                   help="purge archived files in done/ and failed/ older than this many days (0 = keep forever)")
    p.add_argument("--once", action="store_true", help="process the current inbox and exit")
    p.add_argument("--poll", type=float, default=0.0,
                   help="seconds between passes (0 = run once and exit)")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent, send nothing")
    p.add_argument("--require-healthy", action="store_true",
                   help="check GET /health/ready first and exit if the API is not ready")
    p.add_argument("--log", default=None, help="also write logs to this file")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    if args.log:
        handler = logging.FileHandler(args.log)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))
        logging.getLogger().addHandler(handler)

    client = ApiClient(args.api_url, args.api_key, args.connect_timeout,
                       args.read_timeout)

    if args.require_healthy:
        ok, report = client.ready()
        if not ok:
            LOG.error("API at %s is not ready — exiting without touching the inbox",
                      args.api_url)
            if report.get("missing"):
                LOG.error("models not ready: %s", ", ".join(report["missing"]))
            return 0

    if args.dry_run:
        if args.items == "sources":
            pairs, strays = find_sources(Path(args.inbox), args.src_lang,
                                         args.pair_mode)
        else:
            pairs, strays = find_pairs(
                Path(args.inbox), args.src_lang, args.tgt_lang,
                args.pair_mode, translate_unpaired=args.translate_unpaired,
            )
        for pair in pairs:
            if pair.target:
                print(f"would evaluate {pair.source.name} + {pair.target.name}"
                      f"  (key {pair.key})")
            else:
                print(f"would translate+evaluate {pair.source.name}"
                      f" -> {args.target_lang}  (key {pair.key})")
        for path in strays:
            print(f"skip (no counterpart) {path.name}")
        print(f"{len(pairs)} item(s), {len(strays)} unmatched file(s)")
        return 0

    runner = BatchRunner(args, client)
    if args.poll and args.poll > 0:
        LOG.info("watching %s every %ss (Ctrl-C to stop)", args.inbox, args.poll)
        while True:
            try:
                runner.run_once()
            except KeyboardInterrupt:
                LOG.info("stopped")
                return 0
            except Exception:
                LOG.exception("unexpected error in this pass")
            time.sleep(args.poll)
    runner.run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())

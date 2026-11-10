"""
Shared evaluation engine for CompNova's S2ST translation evaluation.
Extracted from DemoApp.py so the same pipeline is used by both the
Streamlit demo and this API service, with no duplicated logic.

Model loaders use functools.lru_cache instead of st.cache_resource,
since this module has no Streamlit dependency - it can be imported by
a plain script, a FastAPI app, or a Streamlit app equally.
"""
import os

os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

import sys
import json
import subprocess
import functools
import hashlib
import logging
import math
import tempfile
import time
import uuid
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

try:
    from .alignment import AlignmentConfig, segment_words, spans, align_units, summarize
except ImportError:
    from alignment import AlignmentConfig, segment_words, spans, align_units, summarize

log = logging.getLogger("uvicorn.error.evaluation")

import numpy as np
import torch

SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

METRICX_REPO_URL = "https://github.com/google-research/metricx.git"
METRICX_PINNED_COMMIT = "fc4978eb064670f7cc33e93ea4f52d38396b8ae6"


def _metricx_directory():
    return Path(os.environ.get("METRICX_DIR", Path(__file__).resolve().parent / "metricx")).resolve()


def _ensure_metricx_repo():
    directory = _metricx_directory()
    if (directory / "metricx24" / "predict.py").exists():
        return
    subprocess.run(["git", "clone", METRICX_REPO_URL, str(directory)],
                   check=True, capture_output=True, text=True, timeout=120)
    subprocess.run(["git", "-C", str(directory), "checkout", METRICX_PINNED_COMMIT],
                   check=True, capture_output=True, text=True, timeout=30)


@functools.lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    return WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)


@functools.lru_cache(maxsize=1)
def get_comet():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    return load_from_checkpoint(model_path)


@functools.lru_cache(maxsize=1)
def get_align_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")


@functools.lru_cache(maxsize=1)
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


def warm_up():
    """Load every model once. Call this at API startup so the FIRST real
    request doesn't pay the cold-load cost (each model can take a while
    to download/load, as seen repeatedly today)."""
    get_whisper()
    get_comet()
    get_align_model()
    get_sentiment_pipeline()


def transcribe_audio(filepath, language=None):
    segments, info = get_whisper().transcribe(
        filepath, language=language, task="transcribe", word_timestamps=True,
        beam_size=5, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500))
    raw = [{"id": i, "text": seg.text, "start": float(seg.start), "end": float(seg.end),
            "avg_logprob": float(seg.avg_logprob), "no_speech_prob": float(seg.no_speech_prob),
            "words": [{"text": w.word, "start": float(w.start), "end": float(w.end),
                       "probability": float(w.probability)} for w in (seg.words or [])]}
           for i, seg in enumerate(segments)]
    return {"segments": raw, "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration), "requested_language": language}


def transcribe_segments(filepath, language=None):
    # Preserve the /transcribe compatibility contract.
    return [s["text"].strip() for s in transcribe_audio(filepath, language)["segments"]]


def join_segment_text(segments):
    return " ".join(segments).strip()


def _encode_spans(units, config):
    model = get_align_model()
    texts = spans(units, config.max_merge)
    # Tokenizer budgets are checked before encoding; never silently truncate.
    texts = {k: v for k, v in texts.items()
             if len(model.tokenizer(v, truncation=False)["input_ids"]) <= model.max_seq_length}
    if not texts:
        return {}
    vectors = model.encode(list(texts.values()), normalize_embeddings=True)
    return dict(zip(texts, vectors))


def cometeval_batch(pairs):
    if not pairs:
        return []
    comet_model = get_comet()
    data = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    use_gpu = 1 if torch.cuda.is_available() else 0
    tokenizer = comet_model.encoder.tokenizer
    limit = min(int(tokenizer.model_max_length), 512)
    for row in data:
        # Conservative combined budget for the QE encoder and special tokens.
        if sum(len(tokenizer(row[k], truncation=False)["input_ids"]) for k in ("src", "mt")) + 4 > limit:
            raise ValueError("COMET input exceeds token budget; reduce max_unit_words/max_merge")
    out = comet_model.predict(data, batch_size=8, gpus=use_gpu)
    return [float(s) for s in out.scores]


def metricx_eval_batch(pairs):
    if not pairs:
        return []
    _ensure_metricx_repo()
    metricx_dir = str(_metricx_directory())
    # Request isolation prevents another worker overwriting this batch's inputs/results.
    with tempfile.TemporaryDirectory(prefix="metricx-") as directory:
        input_path = os.path.join(directory, "input.jsonl")
        output_path = os.path.join(directory, "output.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({"source": p["src"], "hypothesis": p["mt"], "reference": ""}) + "\n")
        command = [sys.executable, "-m", "metricx24.predict", "--tokenizer", "google/mt5-xl",
                   "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6-bfloat16",
                   "--max_input_length", "1536", "--batch_size", "1",
                   "--input_file", input_path, "--output_file", output_path, "--qe"]
        subprocess.run(command, check=True, capture_output=True, cwd=metricx_dir, timeout=600)
        with open(output_path, encoding="utf-8") as f:
            return [float(json.loads(line)["prediction"]) for line in f]


def sentiment_batch(texts):
    if not texts:
        return []
    return get_sentiment_pipeline()(texts)


def evaluate_pair(src_audio_path, mt_audio_path, min_similarity=0.5, max_merge=3,
                  source_language=None, target_language=None, debug=False, config=None):
    config = config or AlignmentConfig(min_similarity=min_similarity, max_merge=max_merge)
    run_id = uuid.uuid4().hex
    timings = {}
    def stage(name, fn):
        start = time.perf_counter()
        log.info("evaluation=%s stage=%s started", run_id, name)
        try:
            result = fn()
        except Exception as exc:
            log.exception("evaluation=%s stage=%s failed", run_id, name)
            error_type = ValueError if isinstance(exc, ValueError) else RuntimeError
            raise error_type(f"evaluation={run_id} stage={name}: {exc}") from exc
        finally:
            timings[name] = time.perf_counter()-start
        log.info("evaluation=%s stage=%s elapsed=%.3f", run_id, name, timings[name])
        return result

    src = stage("source_asr", lambda: transcribe_audio(src_audio_path, source_language))
    tgt = stage("target_asr", lambda: transcribe_audio(mt_audio_path, target_language))
    su = stage("source_segmentation", lambda: segment_words(src["segments"], config))
    tu = stage("target_segmentation", lambda: segment_words(tgt["segments"], config))
    se = stage("source_embeddings", lambda: _encode_spans(su, config)) if su else {}
    te = stage("target_embeddings", lambda: _encode_spans(tu, config)) if tu else {}
    def similarity(i, ni, j, nj):
        a, b = se.get((i, ni)), te.get((j, nj))
        return None if a is None or b is None else float(np.dot(a, b))
    trace = stage("alignment", lambda: align_units(su, tu, similarity, config, debug))
    pairs = []
    for step in trace["path"]:
        if step["operation"] == "match":
            pairs.append({**step, "src": " ".join(su[i]["text"] for i in step["source_ids"]),
                          "mt": " ".join(tu[i]["text"] for i in step["target_ids"])})
    def scores(name, fn):
        def checked():
            values = fn(pairs)
            if len(values) != len(pairs) or not all(math.isfinite(x) for x in values):
                raise ValueError(f"expected {len(pairs)} finite scores, got {len(values)}")
            return values
        return stage(name, checked)
    cs = scores("comet", cometeval_batch)
    ms = scores("metricx", metricx_eval_batch)
    ss = stage("source_sentiment", lambda: sentiment_batch([p["src"] for p in pairs]))
    ts = stage("target_sentiment", lambda: sentiment_batch([p["mt"] for p in pairs]))
    if len(ss) != len(pairs) or len(ts) != len(pairs):
        raise ValueError("Sentiment output count does not match alignment")
    chunks = [{**p, "source": p["src"], "target": p["mt"],
               "source_start": su[p["source_ids"][0]]["start"],
               "source_end": su[p["source_ids"][-1]]["end"],
               "target_start": tu[p["target_ids"][0]]["start"],
               "target_end": tu[p["target_ids"][-1]]["end"], "comet_kiwi": cs[i],
               "metricx_24": ms[i], "source_sentiment": ss[i]["label"],
               "target_sentiment": ts[i]["label"], "sentiment_match": ss[i]["label"] == ts[i]["label"]}
              for i, p in enumerate(pairs)]
    summary = summarize(su, tu, chunks, config)
    unmatched = {"source": [], "target": []}
    for step in trace["path"]:
        if step["operation"] != "match":
            unmatched["source"].extend(su[i]["text"] for i in step["source_ids"])
            unmatched["target"].extend(tu[i]["text"] for i in step["target_ids"])
    summary["unmatched"] = sum(map(len, unmatched.values()))
    summary["sentiment_agreement_pct"] = 100*sum(c["sentiment_match"] for c in chunks)/len(chunks) if chunks else None
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                h.update(block)
        return h.hexdigest()
    packages = {}
    for package in ("faster-whisper", "unbabel-comet", "sentence-transformers", "torch", "numpy"):
        try: packages[package] = version(package)
        except PackageNotFoundError: packages[package] = "not-installed"
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        code_commit = None
    code_hashes = {name: digest(Path(__file__).resolve().parent / name)
                   for name in ("alignment.py", "eval_core.py")}
    return {"schema_version": 2, "run_id": run_id,
            "transcripts": {"source": join_segment_text([s["text"] for s in src["segments"]]),
                            "target": join_segment_text([s["text"] for s in tgt["segments"]])},
            "summary": summary, "chunks": chunks, "unmatched": unmatched,
            "parameters": {**config.to_dict(), "source_language": source_language, "target_language": target_language},
            "provenance": {"code_commit": code_commit, "code_sha256": code_hashes, "source_sha256": digest(src_audio_path), "target_sha256": digest(mt_audio_path),
                           "packages": packages, "models": {"asr": "large-v3-turbo", "alignment": "sentence-transformers/LaBSE",
                           "comet": "Unbabel/wmt22-cometkiwi-da", "metricx": "google/metricx-24-hybrid-large-v2p6-bfloat16"}},
            "trace": {"asr": {"source": src, "target": tgt}, "units": {"source": su, "target": tu},
                      "alignment": trace, "stage_seconds": timings}}

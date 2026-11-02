"""
Shared evaluation engine for CompNova's S2ST translation evaluation.
Used by the FastAPI service (api.py) so the programmatic path and the
dashboard run exactly the same pipeline.

Model loaders use functools.lru_cache instead of st.cache_resource,
since this module has no Streamlit dependency - it can be imported by
a plain script, a FastAPI app, or a Streamlit app equally.

Changes in 2.0 (see API/ROADMAP.md)
-----------------------------------
* ``warm_up()`` no longer raises. It reports per-model status instead, so a
  single failed download cannot stop the service from starting.
* MetricX-24 runs with request-unique file names and is serialised by a
  lock, so two concurrent evaluations can no longer overwrite each other's
  input/output files.
* ``evaluate_pairs_batch()`` evaluates many pairs with ONE MetricX model
  load instead of one per pair. MetricX-24 is executed as a subprocess that
  has to load its model every time, which used to dominate the cost of a
  batch. This is what makes a folder of files affordable.
* Translation (Whisper -> NLLB -> TTS, and direct SeamlessM4T) moved here
  from the Streamlit playground, so the API can serve the whole pipeline.
* Every stage records its duration, so throughput can be reasoned about.
"""
import os

os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

import functools
import json
import logging
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch

import config

LOG = logging.getLogger("compnova.eval.core")

SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)
    except Exception as e:  # pragma: no cover - network/credential issue
        print(f"HF login warning: {e}")

METRICX_REPO_URL = "https://github.com/google-research/metricx.git"
METRICX_PINNED_COMMIT = "fc4978eb064670f7cc33e93ea4f52d38396b8ae6"

# Only one MetricX subprocess at a time: it loads a multi-gigabyte model and
# would otherwise compete for VRAM with the rest of the pipeline.
_metricx_lock = threading.Lock()

# Language maps. The three-letter codes are the ones the playground used.
NLLB_LANG_MAP = {
    "eng": "eng_Latn", "en": "eng_Latn",
    "spa": "spa_Latn", "fra": "fra_Latn", "deu": "deu_Latn", "cmn": "zho_Hans",
    "ita": "ita_Latn", "por": "por_Latn", "jpn": "jpn_Jpan",
}
TTS_LANG_MAP = {
    "eng": "en", "en": "en",
    "spa": "es", "fra": "fr", "deu": "de", "cmn": "zh-CN",
    "ita": "it", "por": "pt", "jpn": "ja",
}
SEAMLESS_LANG_MAP = {
    "eng": "eng", "en": "eng",
    "spa": "spa", "fra": "fra", "deu": "deu", "cmn": "cmn",
    "ita": "ita", "por": "por", "jpn": "jpn",
}
SUPPORTED_LANGUAGES = ("eng", "spa", "fra", "deu", "cmn", "ita", "por", "jpn")

# Per-model load status, surfaced by /health/ready.
MODEL_STATUS: dict[str, dict] = {}


def _record(name: str, ok: bool, seconds: float, error: str | None = None) -> None:
    MODEL_STATUS[name] = {
        "loaded": ok,
        "seconds": round(seconds, 2),
        "error": error,
    }
    if ok:
        LOG.info("model %s ready in %.1fs", name, seconds)
    else:
        LOG.error("model %s failed to load after %.1fs: %s", name, seconds, error)


def _ensure_metricx_repo() -> None:
    """Make sure the pinned MetricX checkout is available.

    The Docker image clones it at build time and the RunPod script links it
    in, so this is normally a no-op. If it is missing we clone it once, but
    we never raise from here - a missing MetricX is reported by
    /health/ready and surfaces as a clean error on the first evaluation.
    """
    target = config.METRICX_PATH
    if os.path.exists(os.path.join(target, "metricx24", "predict.py")):
        return
    try:
        subprocess.run(
            ["git", "clone", METRICX_REPO_URL, target],
            check=True, capture_output=True, text=True, timeout=600,
        )
        subprocess.run(
            ["git", "-C", target, "checkout", METRICX_PINNED_COMMIT],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        LOG.error("could not obtain MetricX at %s: %s", target, exc)


# --------------------------------------------------------------------------- #
# model loaders
# --------------------------------------------------------------------------- #

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
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
    )


@functools.lru_cache(maxsize=1)
def get_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model


@functools.lru_cache(maxsize=1)
def get_seamless():
    from transformers import AutoProcessor, SeamlessM4TModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
    model = SeamlessM4TModel.from_pretrained(
        "facebook/hf-seamless-m4t-medium"
    ).to(device)
    return processor, model, device


_LOADERS = (
    ("whisper", get_whisper),
    ("comet", get_comet),
    ("aligner", get_align_model),
    ("sentiment", get_sentiment_pipeline),
)


def warm_up(include_translation: bool | None = None) -> dict:
    """Load the models once and report what happened.

    Unlike the previous version this NEVER raises: one model that fails to
    download must not stop the service from starting, because then there is
    no /health to report the failure with. Each model is recorded in
    MODEL_STATUS and surfaced by /health/ready.
    """
    started = time.perf_counter()
    loaders = list(_LOADERS)
    if include_translation is None:
        include_translation = config.TRANSLATE_ENABLED
    if include_translation:
        loaders += [("nllb", get_nllb), ("seamless", get_seamless)]

    for name, loader in loaders:
        if MODEL_STATUS.get(name, {}).get("loaded"):
            continue
        t0 = time.perf_counter()
        try:
            loader()
            _record(name, True, time.perf_counter() - t0)
        except Exception as exc:
            _record(name, False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")

    _ensure_metricx_repo()
    metricx_ok = os.path.exists(
        os.path.join(config.METRICX_PATH, "metricx24", "predict.py")
    )
    _record("metricx", metricx_ok, 0.0, None if metricx_ok else
            f"MetricX checkout not found at {config.METRICX_PATH}")

    return model_status()


def model_status() -> dict:
    """A JSON-safe snapshot of every model's state."""
    required = [name for name, _ in _LOADERS] + ["metricx"]
    missing = [n for n in required if not MODEL_STATUS.get(n, {}).get("loaded")]
    return {
        "models": dict(MODEL_STATUS),
        "ready": not missing,
        "missing": missing,
        "total_seconds": round(
            sum(v.get("seconds", 0) for v in MODEL_STATUS.values()), 2
        ),
    }


def models_ready() -> bool:
    return model_status()["ready"]


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #

def transcribe_segments(filepath: str) -> list[str]:
    whisper_model = get_whisper()
    segments, _info = whisper_model.transcribe(
        filepath,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return [seg.text.strip() for seg in segments if seg.text.strip()]


def join_segment_text(segments) -> str:
    return " ".join(segments).strip()


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def _encode_spans(texts, max_merge: int):
    align_model = get_align_model()
    spans, span_texts = [], []
    n = len(texts)
    for length in range(1, max_merge + 1):
        for start in range(0, n - length + 1):
            spans.append((start, length))
            span_texts.append(" ".join(texts[start:start + length]))
    if not span_texts:
        return {}
    embeddings = align_model.encode(span_texts, normalize_embeddings=True)
    return {span: emb for span, emb in zip(spans, embeddings)}


def align_segments_by_content(src_segments, mt_segments, min_similarity: float = 0.5,
                               max_merge: int = 2):
    """Align source and translation segments by MEANING (LaBSE embeddings
    + monotonic dynamic programming), allowing 1-1, 1-2, 2-1, 2-2 merges.
    Segments that can't find an acceptable match are returned separately
    as unmatched, never force-paired or silently dropped."""
    n, m = len(src_segments), len(mt_segments)
    if n == 0 or m == 0:
        return [], list(src_segments), list(mt_segments)

    src_emb = _encode_spans(src_segments, max_merge)
    mt_emb = _encode_spans(mt_segments, max_merge)

    def sim(i, li, j, lj):
        a, b = src_emb.get((i, li)), mt_emb.get((j, lj))
        return None if a is None or b is None else float(np.dot(a, b))

    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, best_bp = NEG, None
            if i > 0 and dp[i - 1][j] > best:
                best, best_bp = dp[i - 1][j], (i - 1, j, 1, 0)
            if j > 0 and dp[i][j - 1] > best:
                best, best_bp = dp[i][j - 1], (i, j - 1, 0, 1)
            for li in range(1, max_merge + 1):
                if i - li < 0:
                    continue
                for lj in range(1, max_merge + 1):
                    if j - lj < 0:
                        continue
                    s = sim(i - li, li, j - lj, lj)
                    if s is None or s < min_similarity:
                        continue
                    cand = dp[i - li][j - lj] + s
                    if cand > best:
                        best, best_bp = cand, (i - li, j - lj, li, lj)
            dp[i][j] = best
            bp[i][j] = best_bp

    pairs, matched_src, matched_mt = [], set(), set()
    i, j = n, m
    while i > 0 or j > 0:
        step = bp[i][j]
        if step is None:
            break
        pi, pj, li, lj = step
        if li > 0 and lj > 0:
            pairs.append({
                "src": " ".join(src_segments[pi:pi + li]),
                "mt": " ".join(mt_segments[pj:pj + lj]),
                "src_count": li, "mt_count": lj,
                "similarity": sim(pi, li, pj, lj),
            })
            matched_src.update(range(pi, pi + li))
            matched_mt.update(range(pj, pj + lj))
        i, j = pi, pj

    pairs.reverse()
    unmatched_src = [src_segments[k] for k in range(n) if k not in matched_src]
    unmatched_mt = [mt_segments[k] for k in range(m) if k not in matched_mt]
    return pairs, unmatched_src, unmatched_mt


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def cometeval_batch(pairs):
    if not pairs:
        return []
    comet_model = get_comet()
    data = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    use_gpu = 1 if torch.cuda.is_available() else 0
    out = comet_model.predict(data, batch_size=8, gpus=use_gpu)
    return [round(float(s), 4) for s in out.scores]


def metricx_eval_batch(pairs):
    """Score pairs with MetricX-24 QE.

    Runs as a subprocess (upstream research code with no library entry
    point). Two consequences are handled explicitly here:

    * the input/output file names are unique per call, so two concurrent
      evaluations cannot overwrite each other's files;
    * the whole call is serialised by a lock, because the subprocess loads a
      multi-gigabyte model and must not run twice at once.
    """
    if not pairs:
        return []

    metricx_dir = config.METRICX_PATH
    if not os.path.exists(os.path.join(metricx_dir, "metricx24", "predict.py")):
        raise RuntimeError(
            f"MODEL_UNAVAILABLE: MetricX checkout not found at {metricx_dir}. "
            "See API/RUNPOD_SETUP.md step 5."
        )

    results_dir = os.path.join(metricx_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    input_path = os.path.abspath(os.path.join(results_dir, f"metricx_in_{token}.jsonl"))
    output_path = os.path.abspath(os.path.join(results_dir, f"metricx_out_{token}.jsonl"))

    with open(input_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "source": p["src"], "hypothesis": p["mt"], "reference": "",
            }) + "\n")

    command = [
        sys.executable, "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6-bfloat16",
        "--max_input_length", "1536",
        "--batch_size", "1",
        "--input_file", input_path,
        "--output_file", output_path,
        "--qe",
    ]
    try:
        with _metricx_lock:
            LOG.info("metricx: scoring %d chunk(s)", len(pairs))
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                cwd=metricx_dir,
                text=True,
                timeout=3600,
            )
        with open(output_path, "r", encoding="utf-8") as f:
            return [
                round(float(json.loads(line).get("prediction", 0.0)), 4)
                for line in f if line.strip()
            ]
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(
            "MetricX-24 failed (exit %s): %s"
            % (exc.returncode, " | ".join(tail) or "no stderr captured")
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MetricX-24 timed out after 60 minutes") from exc
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass


def sentiment_batch(texts):
    if not texts:
        return []
    return get_sentiment_pipeline()(texts)


# --------------------------------------------------------------------------- #
# the evaluation pipeline
# --------------------------------------------------------------------------- #

def _assemble(pairs, unmatched_src, unmatched_mt, comet_scores, metricx_scores,
              src_sent, mt_sent, src_text, mt_text, timings) -> dict:
    chunks = []
    sentiment_matches = []
    for idx, p in enumerate(pairs):
        s_label = src_sent[idx]["label"] if idx < len(src_sent) else None
        m_label = mt_sent[idx]["label"] if idx < len(mt_sent) else None
        match = (s_label == m_label) if s_label and m_label else None
        if match is not None:
            sentiment_matches.append(match)
        chunks.append({
            "source": p["src"],
            "target": p["mt"],
            "comet_kiwi": comet_scores[idx] if idx < len(comet_scores) else None,
            "metricx_24": metricx_scores[idx] if idx < len(metricx_scores) else None,
            "source_sentiment": s_label,
            "target_sentiment": m_label,
            "sentiment_match": match,
        })

    avg_comet = round(sum(comet_scores) / len(comet_scores), 4) if comet_scores else None
    avg_metricx = round(sum(metricx_scores) / len(metricx_scores), 4) if metricx_scores else None
    sentiment_agreement_pct = (
        round(100 * sum(sentiment_matches) / len(sentiment_matches), 1)
        if sentiment_matches else None
    )

    return {
        "transcripts": {"source": src_text, "target": mt_text},
        "summary": {
            "chunks": len(pairs),
            "unmatched": len(unmatched_src) + len(unmatched_mt),
            "comet_kiwi": avg_comet,
            "metricx_24": avg_metricx,
            "sentiment_agreement_pct": sentiment_agreement_pct,
        },
        "chunks": chunks,
        "unmatched": {"source": unmatched_src, "target": unmatched_mt},
        "timings": timings,
    }


def evaluate_pairs_batch(items, min_similarity: float = 0.5, max_merge: int = 2) -> list[dict]:
    """Evaluate many audio pairs, sharing the expensive stages.

    ``items`` is a list of ``{"source_path": str, "target_path": str}``.
    Returns one result dict per item, in the same order.

    Why this exists: MetricX-24 is executed as a subprocess that loads its
    model on every call. Scoring N pairs in one call turns N model loads
    into one. For a folder of 50 files that is the difference between
    roughly an hour of pure overhead and roughly a minute.
    """
    if not items:
        return []

    timings = {"total_s": 0.0}
    t_start = time.perf_counter()

    # ---- 1. transcribe and align ------------------------------------- #
    t0 = time.perf_counter()
    prepared = []
    for item in items:
        src_segments = transcribe_segments(item["source_path"])
        mt_segments = transcribe_segments(item["target_path"])
        pairs, unmatched_src, unmatched_mt = align_segments_by_content(
            src_segments, mt_segments,
            min_similarity=min_similarity, max_merge=max_merge,
        )
        prepared.append({
            "pairs": pairs,
            "unmatched_src": unmatched_src,
            "unmatched_mt": unmatched_mt,
            "src_text": join_segment_text(src_segments),
            "mt_text": join_segment_text(mt_segments),
        })
    timings["transcribe_align_s"] = round(time.perf_counter() - t0, 2)

    # ---- 2. one batch of every chunk from every pair ------------------ #
    flat_pairs = [p for entry in prepared for p in entry["pairs"]]

    t0 = time.perf_counter()
    comet_scores = cometeval_batch(flat_pairs)
    timings["comet_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    metricx_scores = metricx_eval_batch(flat_pairs)
    timings["metricx_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    src_sent = sentiment_batch([p["src"] for p in flat_pairs])
    mt_sent = sentiment_batch([p["mt"] for p in flat_pairs])
    timings["sentiment_s"] = round(time.perf_counter() - t0, 2)

    # ---- 3. hand each pair its own slice of the scores ---------------- #
    results = []
    offset = 0
    for entry in prepared:
        count = len(entry["pairs"])
        results.append(_assemble(
            entry["pairs"],
            entry["unmatched_src"],
            entry["unmatched_mt"],
            comet_scores[offset:offset + count],
            metricx_scores[offset:offset + count],
            src_sent[offset:offset + count],
            mt_sent[offset:offset + count],
            entry["src_text"],
            entry["mt_text"],
            timings,
        ))
        offset += count

    timings["total_s"] = round(time.perf_counter() - t_start, 2)
    for result in results:
        result["timings"] = timings
    return results


def evaluate_pair(src_audio_path: str, mt_audio_path: str,
                  min_similarity: float = 0.5, max_merge: int = 2) -> dict:
    """Evaluate a single pair. Thin wrapper over evaluate_pairs_batch."""
    return evaluate_pairs_batch(
        [{"source_path": src_audio_path, "target_path": mt_audio_path}],
        min_similarity=min_similarity,
        max_merge=max_merge,
    )[0]


# --------------------------------------------------------------------------- #
# translation (moved here from the Streamlit playground)
# --------------------------------------------------------------------------- #

def normalise_language(code: str) -> str:
    """Map a user-supplied language code onto a supported one."""
    if not code:
        return "eng"
    code = code.strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code
    aliases = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu", "zh": "cmn",
               "zh-cn": "cmn", "it": "ita", "pt": "por", "ja": "jpn"}
    return aliases.get(code, code)


def translate_text(source_text: str, target_lang: str) -> str:
    """Text-to-text translation via Meta NLLB-200."""
    tokenizer, model = get_nllb()
    nllb_code = NLLB_LANG_MAP.get(normalise_language(target_lang), "eng_Latn")
    inputs = tokenizer(source_text, return_tensors="pt").to(model.device)
    target_lang_id = tokenizer.lang_code_to_id[nllb_code]
    outputs = model.generate(
        **inputs, forced_bos_token_id=target_lang_id, max_length=512,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def synthesize_speech(text: str, target_lang: str) -> str:
    """Synthesise speech with gTTS. Returns the path to an mp3 file."""
    from gtts import gTTS

    tts_lang = TTS_LANG_MAP.get(normalise_language(target_lang), "en")
    out = f"/tmp/compnova_tts_{uuid.uuid4().hex[:12]}.mp3"
    gTTS(text=text, lang=tts_lang).save(out)
    return out


def _to_wav_16k(src_path: str) -> str:
    """Best-effort conversion to 16 kHz mono WAV using ffmpeg.

    Whisper resamples internally, so this is not strictly required - it just
    makes the synthesised audio behave like a normal input file and keeps
    the container format predictable.
    """
    out = f"/tmp/compnova_tts_{uuid.uuid4().hex[:12]}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
             "-ar", "16000", "-ac", "1", out],
            check=True, capture_output=True, timeout=120,
        )
        return out
    except Exception as exc:
        LOG.warning("ffmpeg conversion failed (%s) - using %s as-is", exc, src_path)
        try:
            os.remove(out)
        except OSError:
            pass
        return src_path


def translate_audio(source_path: str, target_lang: str, mode: str = "cascaded") -> dict:
    """Translate one audio file.

    mode="cascaded": Whisper -> NLLB-200 -> gTTS (transparent, word-level)
    mode="seamless":  SeamlessM4T, one shot from waveform to waveform

    Returns {"audio_path", "text", "mode", "target_lang"}.
    """
    target = normalise_language(target_lang)
    if target == "eng":
        # Nothing to do - but we still need a transcript for the caller.
        segments = transcribe_segments(source_path)
        return {
            "audio_path": source_path,
            "text": join_segment_text(segments),
            "mode": "passthrough",
            "target_lang": target,
        }

    if mode == "seamless":
        processor, model, device = get_seamless()
        import librosa

        audio_array, _ = librosa.load(source_path, sr=16000)
        audio_inputs = processor(
            audio=audio_array, sampling_rate=16000, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang=SEAMLESS_LANG_MAP.get(target, target),
                generate_speech=True,
            )
        import soundfile as sf

        audio = output_tokens[0].cpu().numpy().squeeze()
        out = f"/tmp/compnova_seamless_{uuid.uuid4().hex[:12]}.wav"
        sf.write(out, audio, model.config.sampling_rate)
        segments = transcribe_segments(out)
        return {
            "audio_path": out,
            "text": join_segment_text(segments),
            "mode": "seamless",
            "target_lang": target,
        }

    # cascaded
    segments = transcribe_segments(source_path)
    source_text = join_segment_text(segments)
    translated = translate_text(source_text, target)
    mp3_path = synthesize_speech(translated, target)
    wav_path = _to_wav_16k(mp3_path)
    if wav_path != mp3_path:
        try:
            os.remove(mp3_path)
        except OSError:
            pass
    return {
        "audio_path": wav_path,
        "text": translated,
        "mode": "cascaded",
        "target_lang": target,
        "source_text": source_text,
    }

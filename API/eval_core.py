"""
Shared evaluation engine for CompNova's S2ST translation evaluation.
Used by the FastAPI service (api.py) so the programmatic path and the
dashboard run exactly the same pipeline.

Model loaders use functools.lru_cache instead of st.cache_resource,
since this module has no Streamlit dependency - it can be imported by
a plain script, a FastAPI app, or a Streamlit app equally.

Changes in 2.1 (see API/ROADMAP.md)
-----------------------------------
* ``warm_up()`` no longer raises. It reports per-model status instead, so a
  single failed download cannot stop the service from starting.
* Speech Emotion Recognition (SER) via ``wav2vec2-lg-xlsr-en-speech-emotion-recognition``
  is now integrated into the evaluation pipeline, returning ``ser_match_score``
  (cosine similarity over the 8-class emotion distribution), ``source_emotion_top1``,
  and ``target_emotion_top1`` both at top level and inside ``summary``.
* ``validate_and_normalize_audio()`` verifies audio file integrity, sample rate,
  channel count, and duration, rejecting corrupted files and automatically
  resampling non-16kHz or multi-channel audio to 16 kHz mono WAV.
* MetricX-24 runs with request-unique file names and is serialised by a
  lock, so two concurrent evaluations can no longer overwrite each other's
  input/output files.
* ``evaluate_pairs_batch()`` evaluates many pairs with ONE MetricX model
  load instead of one per pair.
"""
import os

os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

import functools
import json
import logging
import struct
import subprocess
import sys
import threading
import time
import uuid
import wave

import numpy as np
import torch

import config

LOG = logging.getLogger("compnova.eval.core")

SER_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
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
    """Make sure the pinned MetricX checkout is available."""
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
# audio inspection, validation & 16 kHz mono normalization
# --------------------------------------------------------------------------- #

def inspect_audio_file(path: str) -> dict:
    """Inspect an audio file and return its metadata (`sample_rate`, `channels`, `frames`, `duration_s`).

    Raises ValueError if the file is corrupted, unreadable, or has zero duration.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ValueError("audio file is missing or empty")

    # 1. Try soundfile first (fast, supports WAV/FLAC/OGG)
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate and info.samplerate > 0 and info.frames > 0 and info.channels > 0:
            return {
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration_s": float(info.frames) / float(info.samplerate),
            }
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Try standard library wave module for WAV files
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            if sr > 0 and ch > 0 and frames > 0:
                return {
                    "sample_rate": int(sr),
                    "channels": int(ch),
                    "frames": int(frames),
                    "duration_s": float(frames) / float(sr),
                }
    except Exception:
        pass

    # 3. Try ffprobe for compressed containers (MP3, M4A, OPUS, AAC, WMA)
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels,duration:format=duration",
                "-of", "json",
                path,
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        if streams:
            s = streams[0]
            sr = int(s.get("sample_rate") or 0)
            ch = int(s.get("channels") or 0)
            dur = float(s.get("duration") or fmt.get("duration") or 0.0)
            if sr > 0 and ch > 0 and dur > 0:
                return {
                    "sample_rate": sr,
                    "channels": ch,
                    "frames": int(round(sr * dur)),
                    "duration_s": dur,
                }
    except Exception:
        pass

    raise ValueError(f"corrupted or unreadable audio file: {os.path.basename(path)!r}")


def _resample_wav_stdlib(src_path: str, dst_path: str, target_sr: int = 16000,
                         target_channels: int = 1) -> bool:
    """Pure-Python standard-library fallback to convert PCM WAV to 16 kHz mono."""
    try:
        with wave.open(src_path, "rb") as r:
            sr = r.getframerate()
            ch = r.getnchannels()
            sw = r.getsampwidth()
            frames = r.readframes(r.getnFrames() if hasattr(r, "getnFrames") else r.getnframes())
        if sw != 2 or not frames:
            return False
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1)
        if sr != target_sr and len(samples) > 1:
            duration = len(samples) / float(sr)
            new_len = max(1, int(round(duration * target_sr)))
            x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            samples = np.interp(x_new, x_old, samples)
        pcm = np.clip(np.round(samples), -32768, 32767).astype(np.int16).tobytes()
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(target_channels)
            w.setsampwidth(2)
            w.setframerate(target_sr)
            w.writeframes(pcm)
        return True
    except Exception:
        return False


def validate_and_normalize_audio(
    path: str,
    label: str = "audio",
    target_sr: int | None = None,
    target_channels: int | None = None,
    auto_resample: bool | None = None,
) -> dict:
    """Validate audio integrity, sample rate, channel count, and duration.

    When `auto_resample` is True and the audio does not match `target_sr` Hz or
    `target_channels` channels, resamples the file in-place (or to a normalized
    WAV in the same directory) so downstream models receive raw 16 kHz mono
    waveforms.
    """
    target_sr = target_sr or config.EXPECTED_SAMPLE_RATE
    target_channels = target_channels or config.EXPECTED_CHANNELS
    if auto_resample is None:
        auto_resample = config.AUTO_RESAMPLE_16K

    info = inspect_audio_file(path)
    sr = info["sample_rate"]
    channels = info["channels"]
    duration_s = info["duration_s"]

    if duration_s <= 0:
        raise ValueError(f"{label!r} has zero duration")

    needs_conversion = (sr != target_sr) or (channels != target_channels)
    if not needs_conversion:
        return {
            "path": path,
            "sample_rate": sr,
            "channels": channels,
            "duration_s": duration_s,
            "resampled": False,
        }

    if not auto_resample:
        raise ValueError(
            f"{label!r} must be {target_sr} Hz and {target_channels} channel(s), "
            f"got {sr} Hz and {channels} channel(s)"
        )

    base_dir = os.path.dirname(os.path.abspath(path))
    norm_path = os.path.join(base_dir, f"norm16k_{uuid.uuid4().hex[:12]}.wav")
    converted = False

    # 1. Try ffmpeg first
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", path,
                "-ar", str(target_sr), "-ac", str(target_channels),
                "-c:a", "pcm_s16le", norm_path,
            ],
            check=True, capture_output=True, timeout=120,
        )
        converted = os.path.exists(norm_path) and os.path.getsize(norm_path) > 44
    except Exception:
        converted = False

    # 2. Fallback to pure-Python stdlib/numpy resampling for PCM WAV files
    if not converted:
        converted = _resample_wav_stdlib(path, norm_path, target_sr, target_channels)

    # 3. Fallback to librosa + soundfile if installed
    if not converted:
        try:
            import librosa
            import soundfile as sf
            y, _ = librosa.load(path, sr=target_sr, mono=(target_channels == 1))
            sf.write(norm_path, y, target_sr, subtype="PCM_16")
            converted = os.path.exists(norm_path) and os.path.getsize(norm_path) > 44
        except Exception:
            converted = False

    if converted:
        try:
            os.replace(norm_path, path)
            norm_info = inspect_audio_file(path)
            return {
                "path": path,
                "sample_rate": norm_info["sample_rate"],
                "channels": norm_info["channels"],
                "duration_s": norm_info["duration_s"],
                "resampled": True,
                "original_sample_rate": sr,
                "original_channels": channels,
            }
        except Exception:
            if os.path.exists(norm_path):
                try:
                    os.remove(norm_path)
                except OSError:
                    pass

    raise ValueError(
        f"{label!r} has sample_rate={sr} Hz, channels={channels} and could not be "
        f"resampled to {target_sr} Hz mono"
    )


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
def get_ser_pipeline():
    """Load the wav2vec2 Speech Emotion Recognition (SER) model with rebuilt head."""
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_NAME)
    backbone = Wav2Vec2Model.from_pretrained(SER_MODEL_NAME)
    cfg = backbone.config
    dropout_p = getattr(cfg, "final_dropout", 0.1)
    hidden_size = cfg.hidden_size

    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_size, len(SER_LABELS)),
    )

    class _SERModel(nn.Module):
        def __init__(self, bb, clf):
            super().__init__()
            self.backbone = bb
            self.classifier = clf

        def forward(self, input_values, attention_mask=None):
            outputs = self.backbone(input_values, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(pooled)

    model = _SERModel(backbone, classifier)
    weights_path = hf_hub_download(repo_id=SER_MODEL_NAME, filename="model.safetensors")
    state = load_file(weights_path)
    head_weights = {}
    for k, v in state.items():
        if not k.startswith("classifier."):
            continue
        suffix = k[len("classifier."):]
        if suffix.startswith("dense."):
            head_weights["0." + suffix[len("dense."):]] = v
        elif suffix.startswith("output."):
            head_weights["3." + suffix[len("output."):]] = v
    classifier.load_state_dict(head_weights, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    id2label = {i: lbl for i, lbl in enumerate(SER_LABELS)}
    return feature_extractor, model, id2label


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
    """Load the models once and report what happened."""
    loaders = list(_LOADERS)
    if config.SER_ENABLED:
        loaders.append(("ser", get_ser_pipeline))
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
# Speech Emotion Recognition (SER) helpers
# --------------------------------------------------------------------------- #

def do_audio_emotion(audio_path: str) -> list[dict]:
    """Load audio at 16 kHz mono and return sorted 8-class SER predictions."""
    import librosa

    feature_extractor, model, id2label = get_ser_pipeline()
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu()
    preds = [
        {"label": id2label[i], "score": round(float(p), 4)}
        for i, p in enumerate(probs)
    ]
    return sorted(preds, key=lambda x: x["score"], reverse=True)


def compute_affective_match_score(src_preds: list[dict], mt_preds: list[dict]) -> float:
    """Cosine similarity over the canonical 8-class SER label set."""
    if not src_preds or not mt_preds:
        return 0.0
    src_map = {p["label"]: float(p["score"]) for p in src_preds}
    mt_map = {p["label"]: float(p["score"]) for p in mt_preds}
    p = np.array([src_map.get(lbl, 0.0) for lbl in SER_LABELS])
    q = np.array([mt_map.get(lbl, 0.0) for lbl in SER_LABELS])
    p_norm = float(np.linalg.norm(p))
    q_norm = float(np.linalg.norm(q))
    if p_norm == 0.0 or q_norm == 0.0:
        return 0.0
    return round(float(np.dot(p, q) / (p_norm * q_norm)), 4)


def evaluate_ser_pair(src_path: str, mt_path: str) -> dict:
    """Run Speech Emotion Recognition on a source/target audio pair."""
    if not config.SER_ENABLED:
        return {
            "ser_match_score": None,
            "source_emotion_top1": None,
            "target_emotion_top1": None,
            "source_distribution": [],
            "target_distribution": [],
        }
    try:
        src_preds = do_audio_emotion(src_path)
        mt_preds = do_audio_emotion(mt_path)
        score = compute_affective_match_score(src_preds, mt_preds)
        return {
            "ser_match_score": score,
            "source_emotion_top1": src_preds[0]["label"] if src_preds else None,
            "target_emotion_top1": mt_preds[0]["label"] if mt_preds else None,
            "source_distribution": src_preds,
            "target_distribution": mt_preds,
        }
    except Exception as exc:
        LOG.warning("SER evaluation skipped/failed for (%s, %s): %s", src_path, mt_path, exc)
        return {
            "ser_match_score": None,
            "source_emotion_top1": None,
            "target_emotion_top1": None,
            "source_distribution": [],
            "target_distribution": [],
        }


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
    """Score pairs with MetricX-24 QE."""
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
            subprocess.run(
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
              src_sent, mt_sent, src_text, mt_text, timings, ser_info=None) -> dict:
    ser_info = ser_info or {}
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
    ser_match_score = ser_info.get("ser_match_score")
    src_emotion_top1 = ser_info.get("source_emotion_top1")
    tgt_emotion_top1 = ser_info.get("target_emotion_top1")

    return {
        "comet_score": avg_comet,
        "metricx_score": avg_metricx,
        "ser_match_score": ser_match_score,
        "source_emotion_top1": src_emotion_top1,
        "target_emotion_top1": tgt_emotion_top1,
        "transcripts": {"source": src_text, "target": mt_text},
        "summary": {
            "chunks": len(pairs),
            "unmatched": len(unmatched_src) + len(unmatched_mt),
            "comet_kiwi": avg_comet,
            "metricx_24": avg_metricx,
            "sentiment_agreement_pct": sentiment_agreement_pct,
            "ser_match_score": ser_match_score,
            "source_emotion_top1": src_emotion_top1,
            "target_emotion_top1": tgt_emotion_top1,
        },
        "ser": ser_info,
        "chunks": chunks,
        "unmatched": {"source": unmatched_src, "target": unmatched_mt},
        "timings": timings,
    }


def evaluate_pairs_batch(items, min_similarity: float = 0.5, max_merge: int = 2) -> list[dict]:
    """Evaluate many audio pairs, sharing the expensive stages."""
    if not items:
        return []

    timings = {"total_s": 0.0}
    t_start = time.perf_counter()

    # ---- 1. transcribe, align, and run SER --------------------------- #
    t0 = time.perf_counter()
    prepared = []
    for item in items:
        src_segments = transcribe_segments(item["source_path"])
        mt_segments = transcribe_segments(item["target_path"])
        pairs, unmatched_src, unmatched_mt = align_segments_by_content(
            src_segments, mt_segments,
            min_similarity=min_similarity, max_merge=max_merge,
        )
        ser_info = evaluate_ser_pair(item["source_path"], item["target_path"])
        prepared.append({
            "pairs": pairs,
            "unmatched_src": unmatched_src,
            "unmatched_mt": unmatched_mt,
            "src_text": join_segment_text(src_segments),
            "mt_text": join_segment_text(mt_segments),
            "ser_info": ser_info,
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
            ser_info=entry["ser_info"],
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
    """Best-effort conversion to 16 kHz mono WAV using ffmpeg."""
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
    """Translate one audio file."""
    target = normalise_language(target_lang)
    if target == "eng":
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

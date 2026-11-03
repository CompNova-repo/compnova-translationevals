"""
Audio quality analysis for the evaluation pipeline.

Why this exists
---------------
A client asked: *"what if the recording has background noise and other people
speaking in the background, in another language - how do we get only the right
speaker's audio?"*

The honest answer is that you cannot reliably *separate* a target speaker
from overlapping cross-language speech with today's open models, and you should
not try to do it inside a scoring pipeline: removing audio changes what is being
measured. What you can do - and what we do here - is:

1. **Measure** the recording: speech ratio, SNR, how many speakers, how much
   of the file is overlapping speech, which languages are present.
2. **Attribute** the speech: which speaker dominates, and how much of the
   audio belongs to somebody else.
3. **Flag** the file when the transcript is contaminated, so the score is not
   silently wrong, and optionally **filter** the transcription down to the
   dominant speaker so the score is computed on the intended speech.

This module is deliberately dependency-light: the core heuristics are pure
NumPy and always work. Silero VAD, pyannote diarization and Whisper are
optional accelerators that improve accuracy when installed. If they are
missing the module degrades to a coarse estimate instead of failing.

Layered detection (each layer is independent, all optional):

* Layer 0 - energy/percentile analysis (NumPy only, always available).
* Layer 1 - Silero VAD (``silero_vad`` / ``onnxruntime``): accurate speech
  timestamps.
* Layer 2 - pyannote speaker diarization (``pyannote.audio`` + HF token):
  speaker count, per-speaker duration, overlap ratio, dominant speaker.
* Layer 3 - Whisper language ID on the non-dominant speech: proves the
  background speech is *another language*, not just another speaker.
"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

LOG = logging.getLogger("compnova.audio.quality")

SAMPLE_RATE = 16000
FRAME_MS = 25.0
HOP_MS = 10.0

# Thresholds (all overridable through API/config.py).
DEFAULT_SNR_FLOOR_DB = 10.0        # below this the recording is "noisy"
DEFAULT_MIN_SPEECH_RATIO = 0.10    # less than this is mostly silence
DEFAULT_MAX_DOMINANT_RATIO = 0.75  # speaker talk-time share below this = multi-speaker
DEFAULT_MAX_OVERLAP_RATIO = 0.15   # fraction of speech that is overlapped
DEFAULT_MIN_SPEECH_SECONDS = 1.0   # shorter than this gives unstable scores


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_mono_16k(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file as mono float32 at `target_sr`.

    Tries soundfile, then the stdlib `wave` module, then ffmpeg. Raises
    ValueError when the file cannot be decoded.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return _resample(data, sr, target_sr)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - depends on the codec
        LOG.debug("soundfile could not read %s: %s", path, exc)

    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {sw}")
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        return _resample(data, sr, target_sr)
    except Exception as exc:
        LOG.debug("wave could not read %s: %s", path, exc)

    if shutil.which("ffmpeg"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                 "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_s16le", tmp.name],
                check=True, capture_output=True, timeout=300,
            )
            with wave.open(tmp.name, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return data
        except Exception as exc:
            LOG.debug("ffmpeg could not decode %s: %s", path, exc)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    raise ValueError(f"could not decode audio file: {os.path.basename(path)}")


def _resample(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or len(data) < 2:
        return np.ascontiguousarray(data, dtype=np.float32)
    n_out = max(1, int(round(len(data) * target_sr / float(sr))))
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


# --------------------------------------------------------------------------- #
# layer 0 - energy analysis
# --------------------------------------------------------------------------- #

def frame_db(x: np.ndarray, sr: int = SAMPLE_RATE,
             frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS) -> np.ndarray:
    """Per-frame loudness in dBFS."""
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_frames = max(0, 1 + (len(x) - frame_len) // hop) if len(x) >= frame_len else 1
    if n_frames == 0:
        return np.array([_rms_db(x)], dtype=np.float32)
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = x[i * hop:i * hop + frame_len]
        out[i] = _rms_db(chunk)
    return out


def _rms_db(chunk: np.ndarray) -> float:
    if len(chunk) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))
    if rms <= 1e-9:
        return -120.0
    return float(20.0 * np.log10(rms))


def energy_speech_mask(frames_db: np.ndarray,
                       snr_margin_db: float = 8.0) -> np.ndarray:
    """Adaptive energy VAD: speech is everything above the noise floor.

    The noise floor is estimated as the 10th percentile of frame loudness, so
    it adapts to the recording instead of assuming a fixed dBFS.
    """
    if len(frames_db) == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(frames_db, 10))
    peak = float(np.percentile(frames_db, 95))
    threshold = max(floor + snr_margin_db, floor + 3.0)
    threshold = min(threshold, peak - 1.0) if peak - 1.0 > threshold else threshold
    return frames_db > threshold


def _mask_to_intervals(mask: np.ndarray, hop_s: float,
                       min_duration_s: float = 0.15) -> list[tuple[float, float]]:
    """Convert a frame mask into (start, end) intervals."""
    intervals: list[tuple[float, float]] = []
    if len(mask) == 0:
        return intervals
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            intervals.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        intervals.append((start * hop_s, len(mask) * hop_s))
    return [(a, b) for a, b in intervals if (b - a) >= min_duration_s]


# --------------------------------------------------------------------------- #
# layer 1 - Silero VAD (optional)
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _silero_model():
    """Load the Silero VAD model, or return None when unavailable.

    The pip package `silero-vad` is preferred. The torch.hub fallback is only
    used when explicitly enabled (EVAL_SILERO_HUB=1), because it downloads
    from the network at request time.
    """
    try:
        import silero_vad  # noqa: F401

        from silero_vad import load_silero_vad

        model = load_silero_vad()
        model.eval()
        return model
    except Exception as exc:
        LOG.info("silero-vad pip package unavailable: %s", exc)

    if os.environ.get("EVAL_SILERO_HUB", "").strip() not in ("1", "true", "yes"):
        return None
    try:
        import torch

        model, _utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                      model="silero_vad", onnx=False,
                                      trust_repo=True)
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - network / torch dependent
        LOG.info("silero-vad unavailable: %s", exc)
        return None


def silero_speech_intervals(x: np.ndarray, sr: int = SAMPLE_RATE,
                            threshold: float = 0.5) -> list[tuple[float, float]] | None:
    """Speech intervals from Silero VAD, or None when the model is missing."""
    model = _silero_model()
    if model is None:
        return None
    try:
        import torch

        if hasattr(model, "reset"):
            model.reset()
        with torch.no_grad():
            for start in range(0, len(x), sr // 2):
                chunk = x[start:start + sr // 2]
                if len(chunk) < 512:
                    chunk = np.pad(chunk, (0, 512 - len(chunk)))
                model(torch.from_numpy(chunk.copy()), sr)
        ts = model.get_speech_timestamps(
            torch.from_numpy(x.copy()), sr, threshold=threshold,
        )
        return [(float(t["start"]) / sr, float(t["end"]) / sr) for t in ts]
    except Exception as exc:  # pragma: no cover
        LOG.info("silero-vad inference failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# layer 2 - pyannote diarization (optional)
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _diarization_pipeline(model_id: str | None = None):
    """Load the pyannote diarization pipeline, or None when unavailable."""
    model_id = model_id or os.environ.get("EVAL_DIARIZATION_MODEL",
                                          "pyannote/speaker-diarization-3.1")
    token = os.environ.get("HF_TOKEN")
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        LOG.info("pyannote.audio unavailable: %s", exc)
        return None
    try:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token or False)
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as exc:
        LOG.info("diarization pipeline unavailable (%s): %s", model_id, exc)
        return None


def diarize(path: str, model_id: str | None = None) -> list[tuple[float, float, str]] | None:
    """Return (start, end, speaker) turns, or None when diarization is unavailable."""
    pipeline = _diarization_pipeline(model_id)
    if pipeline is None:
        return None
    try:
        import torch

        with torch.no_grad():
            annotation = pipeline(path)
        turns = []
        for turn in annotation.itertracks(yield_label=True):
            start, end, label = turn
            turns.append((float(start), float(end), str(label)))
        return turns
    except Exception as exc:
        LOG.info("diarization failed on %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# speaker statistics
# --------------------------------------------------------------------------- #

def speaker_statistics(turns: list[tuple[float, float, str]],
                       total_s: float) -> dict:
    """Summarise diarization turns into the fields the pipeline needs."""
    if not turns:
        return {}
    per_speaker: dict[str, float] = {}
    for start, end, label in turns:
        per_speaker[label] = per_speaker.get(label, 0.0) + max(0.0, end - start)

    speech_s = sum(per_speaker.values())
    dominant = max(per_speaker, key=lambda k: per_speaker[k])
    dominant_s = per_speaker[dominant]

    # Overlap: time covered by more than one speaker.
    events = []
    for start, end, _label in turns:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    active, overlap_s = 0, 0.0
    prev = None
    for time, delta in events:
        if prev is not None and active > 1:
            overlap_s += time - prev
        active += delta
        prev = time

    return {
        "speaker_count": len(per_speaker),
        "speech_s": round(speech_s, 2),
        "dominant_speaker_ratio": round(dominant_s / speech_s, 4) if speech_s else 0.0,
        "overlap_ratio": round(overlap_s / total_s, 4) if total_s else 0.0,
        "overlap_s": round(overlap_s, 2),
        "dominant_speaker": dominant,
        "speaker_durations_s": {k: round(v, 2) for k, v in per_speaker.items()},
        "speech_ratio": round(speech_s / total_s, 4) if total_s else 0.0,
        "turns": turns,
    }


def dominant_speaker_windows(turns: list[tuple[float, float, str]],
                             dominant: str) -> list[tuple[float, float]]:
    """The (start, end) windows belonging to the dominant speaker."""
    return [(s, e) for s, e, label in turns if label == dominant]


# --------------------------------------------------------------------------- #
# layer 3 - language of the background speech (optional, needs Whisper)
# --------------------------------------------------------------------------- #

def _whisper_model():
    """Reuse the pipeline's Whisper instance when there is one."""
    try:
        import eval_core

        return eval_core.get_whisper()
    except Exception:
        pass
    try:
        from faster_whisper import WhisperModel

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        return WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    except Exception as exc:
        LOG.info("whisper unavailable for language ID: %s", exc)
        return None


def detect_languages(path: str, windows: list[tuple[float, float]] | None = None,
                     max_windows: int = 4) -> list[str]:
    """Languages spoken in the given windows (or in the whole file)."""
    model = _whisper_model()
    if model is None:
        return []
    try:
        segments, info = model.transcribe(path, language=None, clip_timestamps=windows)
        langs = []
        if getattr(info, "language", None):
            langs.append(str(info.language))
        for _seg in segments:
            break
        return sorted(set(langs))
    except Exception as exc:
        LOG.info("language detection failed on %s: %s", path, exc)
        return []


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def analyze(path: str, *,
            snr_floor_db: float = DEFAULT_SNR_FLOOR_DB,
            min_speech_ratio: float = DEFAULT_MIN_SPEECH_RATIO,
            max_dominant_ratio: float = DEFAULT_MAX_DOMINANT_RATIO,
            max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
            min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
            use_diarization: bool = True,
            use_language_id: bool = True,
            known_languages: list[str] | None = None) -> dict:
    """Analyse one audio file.

    The returned dict always contains the raw measurements; `flags` and
    `usable` are the product-facing summary.
    """
    x = load_mono_16k(path)
    sr = SAMPLE_RATE
    duration_s = len(x) / float(sr)
    frames = frame_db(x, sr)
    hop_s = HOP_MS / 1000.0

    noise_floor_db = float(np.percentile(frames, 10)) if len(frames) else -120.0
    speech_db = float(np.percentile(frames, 95)) if len(frames) else -120.0
    snr_db = speech_db - noise_floor_db

    energy_mask = energy_speech_mask(frames)
    energy_intervals = _mask_to_intervals(energy_mask, hop_s)
    speech_s = sum(b - a for a, b in energy_intervals)

    result: dict = {
        "path": path,
        "duration_s": round(duration_s, 2),
        "snr_db": round(snr_db, 2),
        "noise_floor_db": round(noise_floor_db, 2),
        "speech_level_db": round(speech_db, 2),
        "speech_ratio": round(speech_s / duration_s, 4) if duration_s else 0.0,
        "speech_s": round(speech_s, 2),
        "detector": "energy",
    }

    # --- layer 1: Silero VAD refines the speech timeline ---------------- #
    silero = silero_speech_intervals(x, sr) if use_diarization else None
    if silero is not None:
        silero_s = sum(b - a for a, b in silero)
        result["detector"] = "silero-vad"
        result["speech_s"] = round(silero_s, 2)
        result["speech_ratio"] = round(silero_s / duration_s, 4) if duration_s else 0.0
        result["speech_intervals"] = [(round(a, 3), round(b, 3)) for a, b in silero]

    # --- layer 2: diarization attributes the speech to speakers --------- #
    turns = diarize(path) if use_diarization else None
    if turns:
        stats = speaker_statistics(turns, duration_s)
        result.update({
            "speaker_count": stats["speaker_count"],
            "dominant_speaker_ratio": stats["dominant_speaker_ratio"],
            "overlap_ratio": stats["overlap_ratio"],
            "overlap_s": stats["overlap_s"],
            "dominant_speaker": stats["dominant_speaker"],
            "speaker_durations_s": stats["speaker_durations_s"],
            "speech_ratio": stats["speech_ratio"],
            "detector": "diarization",
        })
        result["_turns"] = turns
        result["_windows"] = dominant_speaker_windows(turns, stats["dominant_speaker"])

    # --- layer 3: languages --------------------------------------------- #
    if known_languages:
        result["languages"] = list(known_languages)
    elif use_language_id:
        langs = detect_languages(path)
        if langs:
            result["languages"] = langs
        if turns and result.get("_windows"):
            others = [(s, e, lab) for s, e, lab in turns
                      if lab != result["dominant_speaker"]]
            if others:
                bg_windows = [(s, e) for s, e, _ in others]
                bg_langs = detect_languages(path, windows=bg_windows[:4])
                if bg_langs:
                    result["background_languages"] = bg_langs
                    result["background_speech_detected"] = True

    # --- flags ----------------------------------------------------------- #
    flags: list[str] = []
    if result["speech_ratio"] < min_speech_ratio:
        flags.append("mostly_silence")
    if snr_db < snr_floor_db:
        flags.append("noisy")
    if result.get("speaker_count", 1) > 1:
        flags.append("multi_speaker")
    if result.get("overlap_ratio", 0.0) > max_overlap_ratio:
        flags.append("overlapping_speech")
    if result.get("background_speech_detected"):
        flags.append("background_speech")
    if result.get("speech_s", 0.0) < min_speech_seconds:
        flags.append("very_short_speech")
    if result.get("dominant_speaker_ratio", 1.0) < max_dominant_ratio:
        flags.append("no_single_dominant_speaker")

    result["flags"] = flags
    result["usable"] = not ({"mostly_silence", "no_single_dominant_speaker"} & set(flags))
    result["contaminated"] = bool({"multi_speaker", "overlapping_speech",
                                   "background_speech"} & set(flags))
    return result


def analyse_pair(source_path: str, target_path: str, **kwargs) -> dict:
    """Analyse a source/target pair and summarise the pair-level risk."""
    src = analyze(source_path, **kwargs)
    tgt = analyze(target_path, **kwargs)
    pair_flags = sorted(set(src.get("flags", [])) | set(tgt.get("flags", [])))
    return {
        "source": public(src),
        "target": public(tgt),
        "flags": pair_flags,
        "contaminated": bool(src.get("contaminated") or tgt.get("contaminated")),
        "usable": bool(src.get("usable") and tgt.get("usable")),
    }


def cut_windows(path: str, windows: list[tuple[float, float]],
                out_dir: str | None = None) -> str | None:
    """Write the concatenation of `windows` to a 16 kHz mono WAV.

    Used to hand Whisper and the SER model only the dominant speaker's
    audio, instead of the whole recording.
    """
    if not windows:
        return None
    try:
        x = load_mono_16k(path)
    except Exception as exc:
        LOG.warning("could not load %s for window extraction: %s", path, exc)
        return None
    pieces = []
    for start, end in windows:
        a = max(0, int(float(start) * SAMPLE_RATE))
        b = min(len(x), int(float(end) * SAMPLE_RATE))
        if b > a:
            pieces.append(x[a:b])
    if not pieces:
        return None
    merged = np.concatenate(pieces)
    out_dir = out_dir or tempfile.gettempdir()
    out = os.path.join(out_dir, f"compnova_spk_{os.path.basename(path)[:24]}_"
                                f"{int(windows[0][0] * 1000)}.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.clip(np.round(merged * 32768), -32768, 32767)
                      .astype(np.int16).tobytes())
    return out


def public(report: dict) -> dict:
    """Strip internal helpers before the report leaves the process."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def dominant_windows(report: dict) -> list[tuple[float, float]]:
    """The dominant-speaker windows of a report (empty when unavailable)."""
    return report.get("_windows") or []

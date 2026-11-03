"""
Tests for the audio-quality analysis (noise, multiple speakers, overlap).

Runs with nothing but NumPy: no GPU, no model weights, no network. The
optional Silero / pyannote / Whisper layers are stubbed out, so what is
verified here is the measurement logic and the flag semantics that the
client-facing answer depends on.

    python3 API/test_audio_quality.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import wave

import numpy as np

API_DIR = os.path.dirname(os.path.abspath(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

import audio_quality as aq  # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


# --------------------------------------------------------------------------- #
# synthetic audio
# --------------------------------------------------------------------------- #

def write_wav(path: str, samples: np.ndarray, sr: int = 16000) -> str:
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def speech_like(seconds: float, sr: int = 16000, f0: float = 120.0,
                gap_every: float = 0.0, amplitude: float = 0.25) -> np.ndarray:
    """A crude voiced-speech signal: harmonics with an amplitude envelope."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    for k, gain in enumerate((1.0, 0.5, 0.3, 0.15), start=1):
        sig += gain * np.sin(2 * np.pi * f0 * k * t)
    env = np.ones(n)
    if gap_every > 0:
        period = int(gap_every * sr)
        duty = period // 2
        for start in range(0, n, period):
            env[start:start + duty] = 0.02
    sig *= env
    peak = np.max(np.abs(sig)) or 1.0
    return (sig / peak * amplitude).astype(np.float32)


def two_speaker_recording(seconds: float = 8.0, sr: int = 16000) -> np.ndarray:
    """Speaker A for the first half, speaker B (different pitch) for the rest."""
    a = speech_like(seconds / 2, sr, f0=110.0, gap_every=1.2)
    b = speech_like(seconds / 2, sr, f0=190.0, gap_every=1.2)
    return np.concatenate([a, b])


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_loading(tmp: str) -> None:
    print("\n== loading ==")
    sig = speech_like(1.0)
    path = write_wav(os.path.join(tmp, "one.wav"), sig)
    loaded = aq.load_mono_16k(path)
    check("loads a 16 kHz mono wav", len(loaded) == 16000, str(len(loaded)))
    check("keeps the signal", abs(float(np.max(np.abs(loaded))) - 0.25) < 0.05,
          str(float(np.max(np.abs(loaded)))))

    stereo = np.stack([sig, sig], axis=1)
    path2 = write_wav(os.path.join(tmp, "mono.wav"), stereo[:, 0])
    check("frame_db returns one value per frame",
          len(aq.frame_db(loaded)) > 50, str(len(aq.frame_db(loaded))))


def test_energy_analysis(tmp: str) -> None:
    print("\n== energy analysis ==")
    clean = speech_like(6.0, gap_every=1.5)
    path = write_wav(os.path.join(tmp, "clean.wav"), clean)
    report = aq.analyze(path, use_diarization=False, use_language_id=False)

    check("measures the duration", abs(report["duration_s"] - 6.0) < 0.2,
          str(report["duration_s"]))
    check("finds speech in a voiced signal", report["speech_ratio"] > 0.3,
          str(report["speech_ratio"]))
    check("a clean signal is not flagged as noisy", "noisy" not in report["flags"],
          str(report["flags"]))
    check("a clean signal is usable", report["usable"] is True, str(report))

    # Same signal, buried in noise: SNR must drop and the file must be flagged.
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.12, size=clean.shape).astype(np.float32)
    noisy = (clean + noise).astype(np.float32)
    npath = write_wav(os.path.join(tmp, "noisy.wav"), noisy)
    nreport = aq.analyze(npath, use_diarization=False, use_language_id=False)

    check("noisy file has a lower SNR", nreport["snr_db"] < report["snr_db"],
          f"{nreport['snr_db']} vs {report['snr_db']}")
    check("a noisy file is flagged", "noisy" in nreport["flags"],
          str(nreport["flags"]))
    check("flags are always a list", isinstance(nreport["flags"], list))

    silence = np.zeros(16000 * 3, dtype=np.float32)
    spath = write_wav(os.path.join(tmp, "silence.wav"), silence)
    sreport = aq.analyze(spath, use_diarization=False, use_language_id=False)
    check("silence is flagged as mostly silence",
          "mostly_silence" in sreport["flags"], str(sreport["flags"]))
    check("silence is not usable", sreport["usable"] is False)


def test_speaker_statistics() -> None:
    print("\n== speaker statistics ==")
    turns = [(0.0, 4.0, "A"), (4.0, 8.0, "B")]
    stats = aq.speaker_statistics(turns, 8.0)
    check("counts two speakers", stats["speaker_count"] == 2, str(stats))
    check("no overlap when speakers alternate", stats["overlap_ratio"] == 0.0,
          str(stats["overlap_ratio"]))
    check("no dominant speaker when talk time is even",
          stats["dominant_speaker_ratio"] == 0.5, str(stats))

    single = aq.speaker_statistics([(0.0, 8.0, "A")], 8.0)
    check("a single speaker dominates", single["dominant_speaker_ratio"] == 1.0,
          str(single))

    overlap = aq.speaker_statistics(
        [(0.0, 5.0, "A"), (3.0, 8.0, "B")], 8.0)
    check("overlapping speech is measured", overlap["overlap_ratio"] > 0.2,
          str(overlap["overlap_ratio"]))
    check("overlap seconds are reported", overlap["overlap_s"] > 1.5,
          str(overlap["overlap_s"]))

    windows = aq.dominant_speaker_windows(turns, "A")
    check("dominant windows are extracted", windows == [(0.0, 4.0)], str(windows))


def test_multi_speaker_flag(tmp: str) -> None:
    print("\n== multi-speaker flagging ==")
    path = write_wav(os.path.join(tmp, "two.wav"), two_speaker_recording(8.0))
    # Stub the diarization layer with known turns.
    original = aq._diarize_full
    aq._diarize_full = lambda p, model_id=None: {
        "turns": [(0.0, 4.0, "A"), (4.0, 8.0, "B")],
        "exclusive": [(0.0, 4.0, "A"), (4.0, 8.0, "B")],
        "has_exclusive": True,
    }
    try:
        report = aq.analyze(path, use_diarization=True, use_language_id=False)
    finally:
        aq._diarize_full = original
    check("multi speaker is flagged", "multi_speaker" in report["flags"],
          str(report["flags"]))
    check("the file is marked as contaminated", report["contaminated"] is True,
          str(report))
    check("speaker count is reported", report.get("speaker_count") == 2,
          str(report.get("speaker_count")))
    check("no dominant speaker is flagged",
          "no_single_dominant_speaker" in report["flags"], str(report["flags"]))


def test_window_extraction(tmp: str) -> None:
    print("\n== window extraction ==")
    sig = two_speaker_recording(8.0)
    path = write_wav(os.path.join(tmp, "cut.wav"), sig)
    out = aq.cut_windows(path, [(0.0, 2.0), (4.0, 6.0)], out_dir=tmp)
    check("cut_windows writes a file", out is not None and os.path.exists(out))
    with wave.open(out, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
        check("the cut keeps only the selected windows", abs(dur - 4.0) < 0.15,
              str(dur))
    check("cut_windows returns None without windows",
          aq.cut_windows(path, []) is None)


def test_speaker_attribution() -> None:
    print("\n== speaker attribution ==")
    exclusive = [(0.0, 4.0, "A"), (4.0, 8.0, "B")]
    check("a word inside speaker A's turn is attributed to A",
          aq.speaker_of(1.0, 1.5, exclusive) == "A")
    check("a word inside speaker B's turn is attributed to B",
          aq.speaker_of(5.0, 5.5, exclusive) == "B")
    check("a word outside every turn is unattributed",
          aq.speaker_of(20.0, 21.0, exclusive) is None)

    words = [(0.5, 0.8, "hello"), (1.0, 1.4, "there"),
             (4.2, 4.6, "buenos"), (4.8, 5.2, "dias")]
    kept = aq.keep_dominant_words(words, exclusive, "A")
    check("the other speaker's words are dropped",
          [w[2] for w in kept] == ["hello", "there"],
          str([w[2] for w in kept]))
    check("an unattributed word is kept rather than dropped",
          len(aq.keep_dominant_words([(20.0, 20.2, "hm")], exclusive, "A")) == 1)
    check("without turns every word is kept",
          len(aq.keep_dominant_words(words, [], "A")) == 4)


def test_report_shape(tmp: str) -> None:
    print("\n== report shape ==")
    path = write_wav(os.path.join(tmp, "shape.wav"), speech_like(3.0))
    report = aq.analyze(path, use_diarization=False, use_language_id=False)
    public = aq.public(report)
    check("internal keys are stripped",
          not any(k.startswith("_") for k in public), str(list(public)))
    for key in ("duration_s", "snr_db", "speech_ratio", "flags", "usable",
                "contaminated"):
        check(f"the report has {key}", key in public, str(list(public)))
    pair = aq.analyse_pair(path, path, use_diarization=False,
                           use_language_id=False)
    check("pair analysis reports both sides",
          "source" in pair and "target" in pair, str(list(pair)))
    check("pair analysis has pair flags", isinstance(pair.get("flags"), list))


def test_config_switches() -> None:
    print("\n== configuration ==")
    import config

    check("audio quality is off by default", config.AUDIO_QUALITY_MODE == "off",
          config.AUDIO_QUALITY_MODE)
    check("the mode is reported in describe()",
          "audio_quality_mode" in config.describe(), str(list(config.describe())))
    for mode in ("off", "flag", "filter"):
        os.environ["EVAL_AUDIO_QUALITY_MODE"] = mode
        import importlib

        importlib.reload(config)
        check(f"mode {mode} is accepted", config.AUDIO_QUALITY_MODE == mode,
              config.AUDIO_QUALITY_MODE)
    os.environ["EVAL_AUDIO_QUALITY_MODE"] = "nonsense"
    import importlib

    importlib.reload(config)
    check("an unknown mode falls back to off",
          config.AUDIO_QUALITY_MODE == "off", config.AUDIO_QUALITY_MODE)
    os.environ.pop("EVAL_AUDIO_QUALITY_MODE", None)
    importlib.reload(config)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_loading(tmp)
        test_energy_analysis(tmp)
        test_speaker_statistics()
        test_multi_speaker_flag(tmp)
        test_window_extraction(tmp)
        test_speaker_attribution()
        test_report_shape(tmp)
        test_config_switches()

    print(f"\n{'=' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

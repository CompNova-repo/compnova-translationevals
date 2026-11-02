"""
Shared test scaffolding for the CompNova evaluation service.

Lets the whole HTTP surface be exercised on a machine with no GPU, no model
weights and no network: `torch` is stubbed, every model loader is replaced
with a fake, and the evaluation pipeline returns canned results.

Used by test_api.py (in-process, via FastAPI's TestClient) and test_server.py
(boots a real Uvicorn server and drives it over HTTP).

    pip install fastapi httpx numpy python-multipart
"""
from __future__ import annotations

import os
import struct
import sys
import types

API_DIR = os.path.dirname(os.path.abspath(__file__))
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)


# --------------------------------------------------------------------------- #
# torch stub
# --------------------------------------------------------------------------- #

def install_torch_stub() -> None:
    """Put a minimal `torch` module in sys.modules.

    eval_core only uses torch for cuda availability, dtype selection and
    no_grad; none of that matters when the models are faked.
    """
    if "torch" in sys.modules:
        return
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


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

def configure_env(db_path: str = "/tmp/compnova_test/evaluations.db",
                   **extra) -> None:
    """Point the service at a throwaway database and enable auth."""
    os.environ["EVAL_API_KEYS"] = "test-key-1,test-key-2"
    os.environ["EVAL_DB_PATH"] = db_path
    os.environ["EVAL_SPOOL_DIR"] = os.path.join(os.path.dirname(db_path), "spool")
    os.environ.setdefault("EVAL_JOB_BATCH_SIZE", "4")
    os.environ.setdefault("EVAL_SER_ENABLED", "1")
    for key, value in extra.items():
        os.environ[key] = str(value)


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #

def wav_bytes(seconds: float = 1.0, rate: int = 16000, channels: int = 1) -> bytes:
    """A real, playable RIFF/WAVE file containing silence."""
    frames = int(rate * seconds)
    block = b"\x00\x00" * channels
    data = block * frames
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16
    )
    header += b"data" + struct.pack("<I", len(data))
    return header + data


def wav_claiming_duration(seconds: float, rate: int = 16000) -> bytes:
    """A WAV whose header claims a long duration but carries almost no data.

    Used to test the duration guard without writing a huge file.
    """
    frames = int(rate * seconds)
    data = b"\x00\x00" * 8
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    # Overwrite the RIFF and data sizes so the header claims the long length.
    header = header[:4] + struct.pack("<I", 36 + frames * 2) + header[8:]
    header = header[:40] + struct.pack("<I", frames * 2) + header[44:]
    return header + data


def corrupt_wav_bytes() -> bytes:
    """Bytes with a .wav extension that are not a decodable audio file."""
    return b"RIFF\x10\x00\x00\x00WAVEjunkjunkjunkjunk"


# --------------------------------------------------------------------------- #
# model fakes
# --------------------------------------------------------------------------- #

FAKE_SUMMARY = {
    "chunks": 2, "unmatched": 0, "comet_kiwi": 0.8123,
    "metricx_24": 4.321, "sentiment_agreement_pct": 100.0,
    "ser_match_score": 0.92,
    "source_emotion_top1": "neutral",
    "target_emotion_top1": "calm",
}

FAKE_CHUNKS = [
    {"source": "hello there", "target": "hola ahi", "comet_kiwi": 0.9,
     "metricx_24": 2.104, "source_sentiment": "4 stars",
     "target_sentiment": "4 stars", "sentiment_match": True},
    {"source": "how are you", "target": "como estas", "comet_kiwi": 0.7246,
     "metricx_24": 6.538, "source_sentiment": "4 stars",
     "target_sentiment": "3 stars", "sentiment_match": False},
]


def fake_result() -> dict:
    return {
        "comet_score": 0.8123,
        "metricx_score": 4.321,
        "ser_match_score": 0.92,
        "source_emotion_top1": "neutral",
        "target_emotion_top1": "calm",
        "transcripts": {"source": "hello there how are you",
                        "target": "hola ahi como estas"},
        "summary": dict(FAKE_SUMMARY),
        "ser": {
            "ser_match_score": 0.92,
            "source_emotion_top1": "neutral",
            "target_emotion_top1": "calm",
            "source_distribution": [{"label": "neutral", "score": 0.8}],
            "target_distribution": [{"label": "calm", "score": 0.7}],
        },
        "chunks": [dict(c) for c in FAKE_CHUNKS],
        "unmatched": {"source": [], "target": []},
        "timings": {"total_s": 1.25, "transcribe_align_s": 0.4, "comet_s": 0.2,
                    "metricx_s": 0.5, "sentiment_s": 0.15},
    }


def _probe_audio(source_path, target_path) -> dict:
    """Read the sample rate / channel count the engine actually received."""
    import wave

    info = {}
    for label, path in (("source", source_path), ("target", target_path)):
        try:
            with wave.open(path, "rb") as wf:
                info[f"{label}_sample_rate"] = wf.getframerate()
                info[f"{label}_channels"] = wf.getnchannels()
        except Exception:
            info[f"{label}_sample_rate"] = None
            info[f"{label}_channels"] = None
    return info


class FakeModels:
    """Installs fake model loaders and a fake evaluation pipeline.

    Attributes:
        seen_paths: the source/target paths the pipeline was handed, so a
            test can assert on what actually reached the engine (for example
            that a 8 kHz upload was resampled to 16 kHz mono).
        fail_keys: paths whose content contains a marker string; evaluating
            one of them raises, which is how the poison-pill path is tested.
    """

    def __init__(self):
        self.seen_paths = []
        self.seen_audio = []
        self.fail_keys = []
        self.raise_on_call = False

    def install(self, eval_core) -> "FakeModels":
        def _warm_up(include_translation=None):
            names = ["whisper", "comet", "aligner", "sentiment", "metricx"]
            if eval_core.config.SER_ENABLED:
                names.append("ser")
            for name in names:
                eval_core.MODEL_STATUS[name] = {
                    "loaded": True, "seconds": 0.1, "error": None,
                }
            return eval_core.model_status()

        def _evaluate_pairs_batch(items, min_similarity=0.5, max_merge=2):
            for item in items:
                self.seen_paths.append((item["source_path"], item["target_path"]))
                self.seen_audio.append(_probe_audio(item["source_path"],
                                                    item["target_path"]))
                for key in self.fail_keys:
                    if key in item["source_path"] or key in item["target_path"]:
                        raise RuntimeError(
                            "CUDA out of memory. Tried to allocate 2.00 GiB"
                        )
            if self.raise_on_call:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return [fake_result() for _ in items]

        eval_core.warm_up = _warm_up
        eval_core.evaluate_pairs_batch = _evaluate_pairs_batch
        eval_core.evaluate_pair = (
            lambda src, tgt, min_similarity=0.5, max_merge=2:
            _evaluate_pairs_batch(
                [{"source_path": src, "target_path": tgt}],
                min_similarity=min_similarity, max_merge=max_merge,
            )[0]
        )

        def _translate(source_path, target_lang, mode="cascaded"):
            return {
                "audio_path": source_path,
                "text": f"translated to {target_lang}",
                "source_text": "hello there",
                "mode": mode,
                "target_lang": target_lang,
            }

        eval_core.translate_audio = _translate
        return self


def install_everything(db_path: str = "/tmp/compnova_test/evaluations.db", **extra):
    """Install the stubs, configure the environment, and import the service.

    Returns (api, eval_core, eval_store, FakeModels).
    """
    install_torch_stub()
    configure_env(db_path, **extra)

    import config
    import eval_core
    import eval_store

    fakes = FakeModels().install(eval_core)

    import api

    eval_store.init_db()
    return api, eval_core, eval_store, fakes

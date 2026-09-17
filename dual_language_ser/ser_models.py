"""Single-model multilingual SER inference using emotion2vec+ large.

The same loaded checkpoint and classification head score English and Spanish.
This removes cross-model score incompatibility from the previous design.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from emotion_comparison import EMOTIONS


SER_MODEL_NAME = "iic/emotion2vec_plus_large"
SER_MODEL_HUB = "hf"
EXPECTED_SAMPLE_RATE = 16_000


_LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "fearful": "fearful",
    "fear": "fearful",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "other": "other",
    "sad": "sad",
    "sadness": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
    "unknown": "unknown",
    "unk": "unknown",
    "<unk>": "unknown",
}


def _normalise_label(raw_label: Any) -> str:
    text = str(raw_label).strip().lower()
    # Official token strings are bilingual, e.g. "生气/angry".
    candidates = [part.strip() for part in re.split(r"[/|]", text)]
    candidates.append(text)
    for candidate in candidates:
        cleaned = candidate.strip().strip("<>")
        if candidate in _LABEL_ALIASES:
            return _LABEL_ALIASES[candidate]
        if cleaned in _LABEL_ALIASES:
            return _LABEL_ALIASES[cleaned]
    raise ValueError(f"unsupported emotion2vec label: {raw_label!r}")


def _to_sequence(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        # Some wrapper versions serialize arrays as comma-separated strings.
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
    raise ValueError(f"emotion2vec output field {field_name!r} is not a list")


def _extract_payload(raw_result: Any) -> Mapping[str, Any]:
    result = raw_result
    while isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(
                "expected one emotion2vec result for one audio input, "
                f"received {len(result)}"
            )
        result = result[0]
    if not isinstance(result, Mapping):
        raise ValueError(
            f"unexpected emotion2vec result type: {type(result).__name__}"
        )
    return result


def parse_emotion2vec_output(raw_result: Any) -> dict[str, float]:
    """Parse and normalize FunASR's nine-class emotion2vec output.

    Compatible with the documented ``labels``/``scores`` result and with
    common wrappers that return ``label``/``score``.  A top-1-only response is
    rejected because distribution comparison would otherwise be meaningless.
    """
    payload = _extract_payload(raw_result)
    labels_value = payload.get("labels", payload.get("label"))
    scores_value = payload.get("scores", payload.get("score"))
    if labels_value is None or scores_value is None:
        raise ValueError(
            "emotion2vec result must contain labels/scores; "
            f"available keys: {sorted(map(str, payload.keys()))}"
        )

    labels = _to_sequence(labels_value, "labels")
    scores = _to_sequence(scores_value, "scores")
    if len(labels) != len(scores):
        raise ValueError(
            f"emotion2vec returned {len(labels)} labels but {len(scores)} scores"
        )

    parsed: dict[str, float] = {}
    for raw_label, raw_score in zip(labels, scores):
        label = _normalise_label(raw_label)
        if label in parsed:
            raise ValueError(f"duplicate emotion2vec label after parsing: {label}")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"non-numeric emotion2vec score for {raw_label!r}: {raw_score!r}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"non-finite emotion2vec score for {label}: {score}")
        parsed[label] = score

    missing = set(EMOTIONS) - set(parsed)
    extra = set(parsed) - set(EMOTIONS)
    if missing or extra:
        raise ValueError(
            "emotion2vec must return its complete nine-class distribution; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    values = np.asarray([parsed[label] for label in EMOTIONS], dtype=np.float64)
    # Current FunASR returns posterior scores.  Retain those when valid; use
    # softmax only when a wrapper exposes logits instead.
    if np.any(values < 0.0) or np.any(values > 1.0):
        shifted = values - values.max()
        values = np.exp(shifted)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("emotion2vec scores have zero or invalid total mass")
    values = values / total
    return {label: float(values[index]) for index, label in enumerate(EMOTIONS)}


class SharedMultilingualSER:
    """One loaded model instance used for every language in the pair."""

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = SER_MODEL_NAME,
        hub: str = SER_MODEL_HUB,
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "FunASR is not installed. Run: "
                "python -m pip install -r requirements_ser.txt"
            ) from exc

        funasr_device = "cuda:0" if device == "cuda" else device
        self.model_name = model_name
        self.hub = hub
        self.device = device
        self._model = AutoModel(
            model=model_name,
            hub=hub,
            device=funasr_device,
            disable_update=True,
        )

    def predict_waveform(
        self, waveform: np.ndarray, sampling_rate: int
    ) -> dict[str, float]:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("audio waveform is empty")
        if not np.isfinite(waveform).all():
            raise ValueError("audio waveform contains NaN or infinity")
        raw_result = self._model.generate(
            input=waveform,
            fs=int(sampling_rate),
            granularity="utterance",
            extract_embedding=False,
        )
        return parse_emotion2vec_output(raw_result)

    def predict_file(self, audio_path: str | Path) -> dict[str, float]:
        """Load as mono 16 kHz audio, then run the shared checkpoint."""
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                "librosa is not installed. Run: "
                "python -m pip install -r requirements_ser.txt"
            ) from exc

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        waveform, _ = librosa.load(
            str(path), sr=EXPECTED_SAMPLE_RATE, mono=True, dtype=np.float32
        )
        return self.predict_waveform(waveform, EXPECTED_SAMPLE_RATE)


def load_shared_ser(device: str = "cpu") -> SharedMultilingualSER:
    return SharedMultilingualSER(device=device)

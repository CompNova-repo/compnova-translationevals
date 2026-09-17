"""Comparison utilities for a shared multilingual SER model.

Both audio files must be scored by the same model instance and the same
classification head.  No cross-model label mapping or renormalisation is
performed here.

The returned scores are model posterior scores, not human-calibrated
probabilities.  Using one checkpoint makes the two vectors directly
comparable, but it does not by itself prove equal accuracy across languages.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

NON_SPECIFIC_EMOTIONS = frozenset({"other", "unknown"})


def validate_distribution(
    scores: Mapping[str, float], label: str = "model"
) -> None:
    """Validate a complete nine-class distribution in the model taxonomy."""
    observed = set(scores)
    expected = set(EMOTIONS)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label}: taxonomy mismatch; missing={missing}, extra={extra}"
        )

    values = []
    for emotion in EMOTIONS:
        value = float(scores[emotion])
        if not math.isfinite(value):
            raise ValueError(f"{label}: non-finite score for {emotion!r}")
        if value < -1e-8 or value > 1.0 + 1e-8:
            raise ValueError(
                f"{label}: score outside [0, 1] for {emotion!r}: {value}"
            )
        values.append(value)

    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{label}: scores sum to {total:.8f}, expected 1")


def score_vector(scores: Mapping[str, float]) -> np.ndarray:
    """Return scores in the model's fixed label order."""
    validate_distribution(scores)
    return np.asarray([float(scores[e]) for e in EMOTIONS], dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def js_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - Jensen-Shannon distance, in the range [0, 1]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        return 0.0

    a = a / a.sum()
    b = b / b.sum()
    midpoint = 0.5 * (a + b)

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0.0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    divergence = 0.5 * (
        kl_divergence(a, midpoint) + kl_divergence(b, midpoint)
    )
    distance = math.sqrt(max(0.0, min(1.0, divergence)))
    return float(max(0.0, min(1.0, 1.0 - distance)))


@dataclass(frozen=True)
class EmotionResult:
    language: str
    model: str
    scores: dict[str, float] = field(default_factory=dict)
    top_label: str = ""
    top_score: float = 0.0
    non_specific_mass: float = 0.0


def make_emotion_result(
    scores: Mapping[str, float], language: str, model: str
) -> EmotionResult:
    validate_distribution(scores, label=f"{language} result")
    ordered = {emotion: float(scores[emotion]) for emotion in EMOTIONS}
    top_label = max(EMOTIONS, key=ordered.__getitem__)
    non_specific_mass = math.fsum(
        ordered[label] for label in NON_SPECIFIC_EMOTIONS
    )
    return EmotionResult(
        language=language,
        model=model,
        scores=ordered,
        top_label=top_label,
        top_score=ordered[top_label],
        non_specific_mass=float(non_specific_mass),
    )


@dataclass(frozen=True)
class PairComparison:
    source_path: str
    target_path: str
    source: EmotionResult
    target: EmotionResult
    top_label_agrees: bool
    cosine_similarity: float
    js_similarity: float
    mean_absolute_difference: float
    per_emotion_abs_diff: dict[str, float] = field(default_factory=dict)
    review_reasons: tuple[str, ...] = ()


def compare_pair(
    source: EmotionResult,
    target: EmotionResult,
    source_path: str = "",
    target_path: str = "",
) -> PairComparison:
    """Compare two outputs from the exact same model checkpoint."""
    if source.model != target.model:
        raise ValueError(
            "SER comparison requires the same model checkpoint on both sides: "
            f"{source.model!r} != {target.model!r}"
        )

    source_vector = score_vector(source.scores)
    target_vector = score_vector(target.scores)
    absolute_difference = np.abs(source_vector - target_vector)

    review_reasons: list[str] = []
    if source.top_label in NON_SPECIFIC_EMOTIONS:
        review_reasons.append("SOURCE_TOP_LABEL_NON_SPECIFIC")
    if target.top_label in NON_SPECIFIC_EMOTIONS:
        review_reasons.append("TARGET_TOP_LABEL_NON_SPECIFIC")

    return PairComparison(
        source_path=source_path,
        target_path=target_path,
        source=source,
        target=target,
        top_label_agrees=source.top_label == target.top_label,
        cosine_similarity=cosine_similarity(source_vector, target_vector),
        js_similarity=js_similarity(source_vector, target_vector),
        mean_absolute_difference=float(absolute_difference.mean()),
        per_emotion_abs_diff={
            emotion: float(absolute_difference[index])
            for index, emotion in enumerate(EMOTIONS)
        },
        review_reasons=tuple(review_reasons),
    )


def aggregate(pairs: Iterable[PairComparison]) -> dict:
    items = list(pairs)
    if not items:
        return {
            "pairs_evaluated": 0,
            "top_emotion_agreement_count": 0,
            "top_emotion_agreement_pct": 0.0,
            "cosine_mean": 0.0,
            "cosine_std": 0.0,
            "js_mean": 0.0,
            "js_std": 0.0,
            "mean_absolute_difference_mean": 0.0,
            "review_pair_count": 0,
            "source_top_counts": {},
            "target_top_counts": {},
        }

    cosine = np.asarray(
        [pair.cosine_similarity for pair in items], dtype=np.float64
    )
    js = np.asarray([pair.js_similarity for pair in items], dtype=np.float64)
    mad = np.asarray(
        [pair.mean_absolute_difference for pair in items], dtype=np.float64
    )
    agreement_count = sum(pair.top_label_agrees for pair in items)
    return {
        "pairs_evaluated": len(items),
        "top_emotion_agreement_count": int(agreement_count),
        "top_emotion_agreement_pct": 100.0 * agreement_count / len(items),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(ddof=0)),
        "js_mean": float(js.mean()),
        "js_std": float(js.std(ddof=0)),
        "mean_absolute_difference_mean": float(mad.mean()),
        "review_pair_count": sum(bool(pair.review_reasons) for pair in items),
        "source_top_counts": dict(Counter(p.source.top_label for p in items)),
        "target_top_counts": dict(Counter(p.target.top_label for p in items)),
    }

#!/usr/bin/env python3
"""Sentence-level bilingual speech-emotion comparison.

The pipeline uses word timestamps from one multilingual Whisper model to form
timestamped sentences, aligns source and target sentences with LaBSE, and then
scores every aligned audio span with the same emotion2vec checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from emotion_comparison import EMOTIONS, compare_pair, make_emotion_result
from ser_models import EXPECTED_SAMPLE_RATE, SER_MODEL_NAME, load_shared_ser


ALIGNMENT_MODEL_NAME = "sentence-transformers/LaBSE"
_TERMINAL_PUNCTUATION = re.compile(r"[.!?\u3002\uff01\uff1f]+[\"'\u201d\u2019\u00bb)\]]*$")


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True)
class SentenceUnit:
    index: int
    text: str
    start: float
    end: float
    word_count: int
    mean_word_probability: float | None


@dataclass(frozen=True)
class Alignment:
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    text_similarity: float


def _detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(token.strip() for token in tokens if token.strip()).strip()
    text = re.sub(r"\s+([,.;:!?%\u2026\u3002\uff0c\uff01\uff1f)\]])", r"\1", text)
    text = re.sub(r"([(\[\u00ab])\s+", r"\1", text)
    text = re.sub(r"([\u00bf\u00a1])\s+", r"\1", text)
    return text


def _sentence_from_words(index: int, words: Sequence[WordTimestamp]) -> SentenceUnit:
    probabilities = [
        word.probability
        for word in words
        if word.probability is not None and math.isfinite(word.probability)
    ]
    return SentenceUnit(
        index=index,
        text=_detokenize([word.text for word in words]),
        start=float(words[0].start),
        end=float(words[-1].end),
        word_count=len(words),
        mean_word_probability=(
            float(np.mean(probabilities)) if probabilities else None
        ),
    )


def words_to_sentences(
    words: Sequence[WordTimestamp],
    max_seconds: float = 20.0,
    max_words: int = 50,
) -> list[SentenceUnit]:
    """Group timestamped ASR words at inferred sentence punctuation.

    Long unpunctuated ASR runs are split at configurable duration/word limits,
    which prevents a punctuation error from creating an unusably large span.
    """
    if max_seconds <= 0.0:
        raise ValueError("max_seconds must be positive")
    if max_words <= 0:
        raise ValueError("max_words must be positive")

    sentences: list[SentenceUnit] = []
    pending: list[WordTimestamp] = []
    previous_start = -math.inf

    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if not (math.isfinite(word.start) and math.isfinite(word.end)):
            raise ValueError("word timestamps must be finite")
        if word.start < 0.0 or word.end <= word.start:
            raise ValueError(f"invalid word timestamp: {word}")
        if word.start < previous_start:
            raise ValueError("word timestamps must be in chronological order")
        previous_start = word.start
        pending.append(word)

        elapsed = pending[-1].end - pending[0].start
        is_terminal = bool(_TERMINAL_PUNCTUATION.search(text))
        if is_terminal or elapsed >= max_seconds or len(pending) >= max_words:
            sentences.append(_sentence_from_words(len(sentences), pending))
            pending = []

    if pending:
        sentences.append(_sentence_from_words(len(sentences), pending))
    return sentences


def transcribe_words(
    model: Any,
    audio_path: Path,
    language: str | None,
    max_sentence_seconds: float,
    max_sentence_words: int,
) -> tuple[list[SentenceUnit], dict[str, Any]]:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    words: list[WordTimestamp] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if not segment_words:
            continue
        for word in segment_words:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if start is None or end is None:
                continue
            probability = getattr(word, "probability", None)
            words.append(
                WordTimestamp(
                    text=str(getattr(word, "word", "")),
                    start=float(start),
                    end=float(end),
                    probability=(
                        float(probability) if probability is not None else None
                    ),
                )
            )

    sentences = words_to_sentences(
        words,
        max_seconds=max_sentence_seconds,
        max_words=max_sentence_words,
    )
    if not sentences:
        raise RuntimeError(f"ASR returned no timestamped speech for {audio_path}")

    metadata = {
        "requested_language": language,
        "detected_language": getattr(info, "language", None),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
        "word_count": len(words),
        "sentence_count": len(sentences),
    }
    return sentences, metadata


def _merged_text(sentences: Sequence[SentenceUnit], start: int, end: int) -> str:
    return " ".join(sentence.text for sentence in sentences[start:end]).strip()


def _span_embeddings(
    sentences: Sequence[SentenceUnit], model: Any, max_merge: int
) -> dict[tuple[int, int], np.ndarray]:
    keys: list[tuple[int, int]] = []
    texts: list[str] = []
    for start in range(len(sentences)):
        for width in range(1, max_merge + 1):
            end = start + width
            if end <= len(sentences):
                keys.append((start, end))
                texts.append(_merged_text(sentences, start, end))

    if not texts:
        return {}
    encoded = np.asarray(
        model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float64,
    )
    if encoded.ndim != 2 or encoded.shape[0] != len(keys):
        raise RuntimeError("alignment model returned an unexpected embedding shape")
    return {key: encoded[index] for index, key in enumerate(keys)}


def monotonic_align(
    source: Sequence[SentenceUnit],
    target: Sequence[SentenceUnit],
    model: Any,
    max_merge: int = 2,
    min_similarity: float = 0.50,
    skip_penalty: float = -0.15,
) -> list[Alignment]:
    """Align sentence spans with order-preserving dynamic programming."""
    if max_merge <= 0:
        raise ValueError("max_merge must be positive")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [-1, 1]")
    if skip_penalty > 0.0:
        raise ValueError("skip_penalty must be zero or negative")
    if not source or not target:
        return []

    source_embeddings = _span_embeddings(source, model, max_merge)
    target_embeddings = _span_embeddings(target, model, max_merge)
    source_count = len(source)
    target_count = len(target)
    scores = np.full((source_count + 1, target_count + 1), -np.inf)
    scores[0, 0] = 0.0
    back: dict[tuple[int, int], tuple[int, int, Alignment | None]] = {}

    def update(
        next_i: int,
        next_j: int,
        score: float,
        previous_i: int,
        previous_j: int,
        alignment: Alignment | None,
    ) -> None:
        if score > scores[next_i, next_j] + 1e-12:
            scores[next_i, next_j] = score
            back[next_i, next_j] = (previous_i, previous_j, alignment)

    for i in range(source_count + 1):
        for j in range(target_count + 1):
            current = float(scores[i, j])
            if not math.isfinite(current):
                continue

            # Try matches before skips so exact-score ties prefer an alignment.
            for source_width in range(1, max_merge + 1):
                source_end = i + source_width
                if source_end > source_count:
                    break
                source_vector = source_embeddings[(i, source_end)]
                for target_width in range(1, max_merge + 1):
                    target_end = j + target_width
                    if target_end > target_count:
                        break
                    target_vector = target_embeddings[(j, target_end)]
                    similarity = float(np.dot(source_vector, target_vector))
                    similarity = max(-1.0, min(1.0, similarity))
                    if similarity < min_similarity:
                        continue
                    alignment = Alignment(
                        source_start=i,
                        source_end=source_end,
                        target_start=j,
                        target_end=target_end,
                        text_similarity=similarity,
                    )
                    update(
                        source_end,
                        target_end,
                        current + similarity,
                        i,
                        j,
                        alignment,
                    )

            if i < source_count:
                update(i + 1, j, current + skip_penalty, i, j, None)
            if j < target_count:
                update(i, j + 1, current + skip_penalty, i, j, None)

    alignments: list[Alignment] = []
    cursor = (source_count, target_count)
    while cursor != (0, 0):
        step = back.get(cursor)
        if step is None:
            raise RuntimeError("alignment backtracking failed")
        previous_i, previous_j, alignment = step
        if alignment is not None:
            alignments.append(alignment)
        cursor = (previous_i, previous_j)
    alignments.reverse()
    return alignments


def extract_audio_span(
    waveform: np.ndarray,
    start: float,
    end: float,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    padding_seconds: float = 0.15,
    minimum_seconds: float = 0.30,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("audio waveform is empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start < 0.0 or end <= start:
        raise ValueError(f"invalid audio span: start={start}, end={end}")
    if padding_seconds < 0.0 or minimum_seconds <= 0.0:
        raise ValueError("padding_seconds and minimum_seconds must be valid")

    total_seconds = waveform.size / sample_rate
    if start >= total_seconds:
        raise ValueError(f"audio span starts after the waveform ends: {start:.3f}s")
    padded_start = max(0.0, start - padding_seconds)
    padded_end = min(total_seconds, end + padding_seconds)
    if padded_end - padded_start < minimum_seconds:
        centre = 0.5 * (padded_start + padded_end)
        padded_start = max(0.0, centre - minimum_seconds / 2.0)
        padded_end = min(total_seconds, padded_start + minimum_seconds)
        padded_start = max(0.0, padded_end - minimum_seconds)

    first_sample = max(0, int(math.floor(padded_start * sample_rate)))
    last_sample = min(waveform.size, int(math.ceil(padded_end * sample_rate)))
    chunk = waveform[first_sample:last_sample]
    if chunk.size == 0:
        raise ValueError("audio span produced an empty waveform")
    return chunk


def _span_details(
    sentences: Sequence[SentenceUnit], start: int, end: int
) -> dict[str, Any]:
    selected = sentences[start:end]
    probabilities = [
        sentence.mean_word_probability
        for sentence in selected
        if sentence.mean_word_probability is not None
    ]
    span_start = float(selected[0].start)
    span_end = float(selected[-1].end)
    return {
        "sentence_indices": list(range(start, end)),
        "text": _merged_text(sentences, start, end),
        "start": span_start,
        "end": span_end,
        "duration": span_end - span_start,
        "mean_word_probability": (
            float(np.mean(probabilities)) if probabilities else None
        ),
    }


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    weight_array = np.asarray(weights, dtype=np.float64)
    if np.any(weight_array < 0.0) or weight_array.sum() <= 0.0:
        raise ValueError("aggregation weights must have positive total mass")
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weight_array))


def aggregate_sentence_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {
            "aligned_pair_count": 0,
            "total_weighted_seconds": 0.0,
            "duration_weighted_cosine_similarity": 0.0,
            "duration_weighted_js_similarity": 0.0,
            "duration_weighted_top_label_agreement_pct": 0.0,
            "cosine_similarity_mean": 0.0,
            "js_similarity_mean": 0.0,
            "top_label_agreement_pct": 0.0,
        }

    weights = [float(pair["weight_seconds"]) for pair in pairs]
    cosine = [float(pair["emotion"]["cosine_similarity"]) for pair in pairs]
    js_values = [float(pair["emotion"]["js_similarity"]) for pair in pairs]
    agreement = [
        1.0 if pair["emotion"]["top_label_agrees"] else 0.0 for pair in pairs
    ]
    return {
        "aligned_pair_count": len(pairs),
        "total_weighted_seconds": float(math.fsum(weights)),
        "duration_weighted_cosine_similarity": _weighted_mean(cosine, weights),
        "duration_weighted_js_similarity": _weighted_mean(js_values, weights),
        "duration_weighted_top_label_agreement_pct": 100.0
        * _weighted_mean(agreement, weights),
        "cosine_similarity_mean": float(np.mean(cosine)),
        "js_similarity_mean": float(np.mean(js_values)),
        "top_label_agreement_pct": 100.0 * float(np.mean(agreement)),
    }


def _clear_model_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _load_audio(path: Path) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is not installed. Run: "
            "python -m pip install -r requirements_ser.txt"
        ) from exc
    waveform, _ = librosa.load(
        str(path), sr=EXPECTED_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if waveform.size == 0:
        raise ValueError(f"audio file is empty: {path}")
    return waveform


def _unmatched_indices(
    sentence_count: int, alignments: Sequence[Alignment], side: str
) -> list[int]:
    matched: set[int] = set()
    for alignment in alignments:
        start = getattr(alignment, f"{side}_start")
        end = getattr(alignment, f"{side}_end")
        matched.update(range(start, end))
    return sorted(set(range(sentence_count)) - matched)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_file).expanduser().resolve()
    target_path = Path(args.target_file).expanduser().resolve()
    for path in (source_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")

    device = resolve_device(args.device)
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: "
            "python -m pip install -r requirements_ser.txt"
        ) from exc

    print(f"Loading multilingual ASR model {args.whisper_model!r} on {device}...")
    asr_model = WhisperModel(
        args.whisper_model, device=device, compute_type=compute_type
    )
    source_sentences, source_asr = transcribe_words(
        asr_model,
        source_path,
        args.source_language,
        args.max_sentence_seconds,
        args.max_sentence_words,
    )
    target_sentences, target_asr = transcribe_words(
        asr_model,
        target_path,
        args.target_language,
        args.max_sentence_seconds,
        args.max_sentence_words,
    )
    del asr_model
    _clear_model_memory(device)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "python -m pip install -r requirements_ser.txt"
        ) from exc

    print(f"Aligning {len(source_sentences)} source and {len(target_sentences)} target sentences...")
    alignment_model = SentenceTransformer(ALIGNMENT_MODEL_NAME, device=device)
    alignments = monotonic_align(
        source_sentences,
        target_sentences,
        alignment_model,
        max_merge=args.max_merge,
        min_similarity=args.min_alignment_similarity,
        skip_penalty=args.skip_penalty,
    )
    del alignment_model
    _clear_model_memory(device)
    if not alignments:
        raise RuntimeError(
            "no sentence pairs met the alignment threshold; inspect the ASR "
            "output or lower --min-alignment-similarity"
        )

    source_audio = _load_audio(source_path)
    target_audio = _load_audio(target_path)
    print(f"Scoring {len(alignments)} aligned spans with {SER_MODEL_NAME}...")
    ser_model = load_shared_ser(device=device)

    pair_results: list[dict[str, Any]] = []
    for pair_index, alignment in enumerate(alignments):
        source_span = _span_details(
            source_sentences, alignment.source_start, alignment.source_end
        )
        target_span = _span_details(
            target_sentences, alignment.target_start, alignment.target_end
        )
        source_chunk = extract_audio_span(
            source_audio,
            source_span["start"],
            source_span["end"],
            padding_seconds=args.padding_seconds,
            minimum_seconds=args.minimum_slice_seconds,
        )
        target_chunk = extract_audio_span(
            target_audio,
            target_span["start"],
            target_span["end"],
            padding_seconds=args.padding_seconds,
            minimum_seconds=args.minimum_slice_seconds,
        )

        source_scores = ser_model.predict_waveform(
            source_chunk, EXPECTED_SAMPLE_RATE
        )
        target_scores = ser_model.predict_waveform(
            target_chunk, EXPECTED_SAMPLE_RATE
        )
        source_emotion = make_emotion_result(
            source_scores, args.source_language, SER_MODEL_NAME
        )
        target_emotion = make_emotion_result(
            target_scores, args.target_language, SER_MODEL_NAME
        )
        comparison = compare_pair(source_emotion, target_emotion)

        review_reasons = list(comparison.review_reasons)
        if alignment.text_similarity < args.review_alignment_similarity:
            review_reasons.append("LOW_TEXT_ALIGNMENT_SIMILARITY")
        for side, span in (("SOURCE", source_span), ("TARGET", target_span)):
            confidence = span["mean_word_probability"]
            if confidence is not None and confidence < args.review_asr_probability:
                review_reasons.append(f"LOW_{side}_ASR_PROBABILITY")

        pair_results.append(
            {
                "pair_index": pair_index,
                "text_similarity": alignment.text_similarity,
                "weight_seconds": 0.5
                * (source_span["duration"] + target_span["duration"]),
                "source": source_span,
                "target": target_span,
                "emotion": {
                    "source_top_label": source_emotion.top_label,
                    "source_top_score": source_emotion.top_score,
                    "target_top_label": target_emotion.top_label,
                    "target_top_score": target_emotion.top_score,
                    "top_label_agrees": comparison.top_label_agrees,
                    "cosine_similarity": comparison.cosine_similarity,
                    "js_similarity": comparison.js_similarity,
                    "mean_absolute_difference": comparison.mean_absolute_difference,
                    "source_scores": source_emotion.scores,
                    "target_scores": target_emotion.scores,
                    "per_emotion_abs_diff": comparison.per_emotion_abs_diff,
                },
                "review_reasons": review_reasons,
            }
        )

    result = {
        "schema_version": 1,
        "models": {
            "asr": args.whisper_model,
            "alignment": ALIGNMENT_MODEL_NAME,
            "ser": SER_MODEL_NAME,
        },
        "settings": {
            "device": device,
            "asr_compute_type": compute_type,
            "max_sentence_seconds": args.max_sentence_seconds,
            "max_sentence_words": args.max_sentence_words,
            "max_merge": args.max_merge,
            "min_alignment_similarity": args.min_alignment_similarity,
            "skip_penalty": args.skip_penalty,
            "padding_seconds": args.padding_seconds,
            "minimum_slice_seconds": args.minimum_slice_seconds,
        },
        "source": {
            "path": str(source_path),
            "language": args.source_language,
            "asr": source_asr,
            "sentences": [asdict(sentence) for sentence in source_sentences],
            "unmatched_sentence_indices": _unmatched_indices(
                len(source_sentences), alignments, "source"
            ),
        },
        "target": {
            "path": str(target_path),
            "language": args.target_language,
            "asr": target_asr,
            "sentences": [asdict(sentence) for sentence in target_sentences],
            "unmatched_sentence_indices": _unmatched_indices(
                len(target_sentences), alignments, "target"
            ),
        },
        "summary": aggregate_sentence_pairs(pair_results),
        "pairs": pair_results,
    }
    return result


def write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sentence_ser_results.json"
    csv_path = output_dir / "sentence_ser_results.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "pair_index",
        "source_sentence_indices",
        "target_sentence_indices",
        "source_start",
        "source_end",
        "target_start",
        "target_end",
        "source_text",
        "target_text",
        "text_similarity",
        "weight_seconds",
        "source_top_label",
        "source_top_score",
        "target_top_label",
        "target_top_score",
        "top_label_agrees",
        "cosine_similarity",
        "js_similarity",
        "mean_absolute_difference",
        "review_reasons",
    ]
    fields.extend(f"source_{emotion}" for emotion in EMOTIONS)
    fields.extend(f"target_{emotion}" for emotion in EMOTIONS)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in result["pairs"]:
            emotion = pair["emotion"]
            row = {
                "pair_index": pair["pair_index"],
                "source_sentence_indices": ";".join(
                    map(str, pair["source"]["sentence_indices"])
                ),
                "target_sentence_indices": ";".join(
                    map(str, pair["target"]["sentence_indices"])
                ),
                "source_start": pair["source"]["start"],
                "source_end": pair["source"]["end"],
                "target_start": pair["target"]["start"],
                "target_end": pair["target"]["end"],
                "source_text": pair["source"]["text"],
                "target_text": pair["target"]["text"],
                "text_similarity": pair["text_similarity"],
                "weight_seconds": pair["weight_seconds"],
                "source_top_label": emotion["source_top_label"],
                "source_top_score": emotion["source_top_score"],
                "target_top_label": emotion["target_top_label"],
                "target_top_score": emotion["target_top_score"],
                "top_label_agrees": emotion["top_label_agrees"],
                "cosine_similarity": emotion["cosine_similarity"],
                "js_similarity": emotion["js_similarity"],
                "mean_absolute_difference": emotion[
                    "mean_absolute_difference"
                ],
                "review_reasons": ";".join(pair["review_reasons"]),
            }
            for label in EMOTIONS:
                row[f"source_{label}"] = emotion["source_scores"][label]
                row[f"target_{label}"] = emotion["target_scores"][label]
            writer.writerow(row)
    return json_path, csv_path


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _FakeEmbeddingModel:
    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        vectors = {
            "hello world": [1.0, 0.0, 0.0],
            "hola": [0.0, 0.0, 1.0],
            "mundo": [0.0, 0.0, -1.0],
            "hola mundo": [1.0, 0.0, 0.0],
            "bye": [0.0, 1.0, 0.0],
            "adios": [0.0, 1.0, 0.0],
            "hello world bye": [0.0, -1.0, 0.0],
            "mundo adios": [0.0, 0.0, -1.0],
        }
        array = np.asarray([vectors[text] for text in texts], dtype=np.float64)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / norms


def run_self_test() -> None:
    words = [
        WordTimestamp("Hello", 0.0, 0.2, 0.9),
        WordTimestamp("world.", 0.2, 0.5, 0.8),
        WordTimestamp("How", 0.7, 0.9, 0.9),
        WordTimestamp("are", 0.9, 1.1, 0.8),
        WordTimestamp("you?", 1.1, 1.4, 0.7),
    ]
    sentences = words_to_sentences(words)
    assert [sentence.text for sentence in sentences] == [
        "Hello world.",
        "How are you?",
    ]
    assert sentences[0].start == 0.0 and sentences[0].end == 0.5

    source = [
        SentenceUnit(0, "hello world", 0.0, 1.0, 2, 0.9),
        SentenceUnit(1, "bye", 1.1, 1.5, 1, 0.9),
    ]
    target = [
        SentenceUnit(0, "hola", 0.0, 0.4, 1, 0.9),
        SentenceUnit(1, "mundo", 0.4, 0.9, 1, 0.9),
        SentenceUnit(2, "adios", 1.0, 1.4, 1, 0.9),
    ]
    alignments = monotonic_align(
        source,
        target,
        _FakeEmbeddingModel(),
        max_merge=2,
        min_similarity=0.8,
    )
    assert [(a.source_end - a.source_start, a.target_end - a.target_start) for a in alignments] == [
        (1, 2),
        (1, 1),
    ]

    waveform = np.arange(2 * EXPECTED_SAMPLE_RATE, dtype=np.float32)
    chunk = extract_audio_span(
        waveform,
        start=0.0,
        end=0.1,
        padding_seconds=0.0,
        minimum_seconds=0.3,
    )
    assert chunk.size == int(0.3 * EXPECTED_SAMPLE_RATE)

    pairs = [
        {
            "weight_seconds": 1.0,
            "emotion": {
                "cosine_similarity": 0.5,
                "js_similarity": 0.25,
                "top_label_agrees": False,
            },
        },
        {
            "weight_seconds": 3.0,
            "emotion": {
                "cosine_similarity": 1.0,
                "js_similarity": 0.75,
                "top_label_agrees": True,
            },
        },
    ]
    summary = aggregate_sentence_pairs(pairs)
    assert math.isclose(summary["duration_weighted_cosine_similarity"], 0.875)
    assert math.isclose(summary["duration_weighted_js_similarity"], 0.625)
    assert math.isclose(
        summary["duration_weighted_top_label_agreement_pct"], 75.0
    )
    print("Self-test passed: sentence grouping, alignment, slicing, and aggregation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align bilingual speech sentence by sentence and compare each "
            "pair with one shared emotion2vec SER model."
        )
    )
    parser.add_argument("--source-file")
    parser.add_argument("--target-file")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", default="es")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-type",
        help="CTranslate2 compute type; defaults to float16 on CUDA and int8 on CPU",
    )
    parser.add_argument("--max-sentence-seconds", type=float, default=20.0)
    parser.add_argument("--max-sentence-words", type=int, default=50)
    parser.add_argument("--max-merge", type=int, choices=(1, 2), default=2)
    parser.add_argument("--min-alignment-similarity", type=float, default=0.50)
    parser.add_argument("--review-alignment-similarity", type=float, default=0.65)
    parser.add_argument("--skip-penalty", type=float, default=-0.15)
    parser.add_argument("--review-asr-probability", type=float, default=0.50)
    parser.add_argument("--padding-seconds", type=float, default=0.15)
    parser.add_argument("--minimum-slice-seconds", type=float, default=0.30)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sentence_ser")
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.source_file or not args.target_file:
        parser.error("--source-file and --target-file are required")

    result = evaluate(args)
    json_path, csv_path = write_outputs(result, args.output_dir)
    summary = result["summary"]
    print(
        f"Done: {summary['aligned_pair_count']} aligned pairs; "
        f"duration-weighted cosine={summary['duration_weighted_cosine_similarity']:.4f}; "
        f"duration-weighted JS={summary['duration_weighted_js_similarity']:.4f}"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()

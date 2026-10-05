#!/usr/bin/env python3
"""
Build one ~2-minute English/Spanish DRAL evaluation pair.

SAFETY PRINCIPLES
-----------------
1. Never pair EN/ES files based only on directory sorting.
2. Only use explicitly supplied/verified pairs.
3. Verify every file exists.
4. Verify every pair contains exactly one EN and one ES file.
5. Reject duplicate usage.
6. Keep pair order identical in both output recordings.
7. Add the same silence between corresponding utterances.
8. Produce a JSON manifest showing exactly what was merged.

Recommended workflow:
    First create verified_pairs.csv from DRAL metadata / known verified pairs.

verified_pairs.csv:

pair_id,en_file,es_file
001_010,EN_001_2.10.wav,ES_001_1.10.wav
001_011,EN_001_2.11.wav,ES_001_1.11.wav
001_012,EN_001_2.12.wav,ES_001_1.12.wav

Then:

python build_dral_pair.py \
    --audio-dir /path/to/DRAL/fragments-short \
    --pairs verified_pairs.csv \
    --output-dir output \
    --target-seconds 120
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_SR = 16000


@dataclass(frozen=True)
class Pair:
    pair_id: str
    en_path: Path
    es_path: Path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read 16-bit PCM WAV, downmixing multi-channel/stereo to mono if needed."""

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"{path.name}: expected 16-bit PCM WAV, "
            f"got sample width={sample_width}"
        )

    audio = np.frombuffer(frames, dtype="<i2").copy()

    if len(audio) == 0:
        raise ValueError(f"{path.name}: empty audio file")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype("<i2")

    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype="<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def duration(audio: np.ndarray, sr: int) -> float:
    return len(audio) / sr


def load_pairs(csv_path: Path, audio_dir: Path) -> list[Pair]:
    """
    Load EXPLICIT pair mappings.

    This intentionally does NOT try to guess EN ↔ ES relationships
    from filenames.
    """

    required_columns = {"pair_id", "en_file", "es_file"}

    pairs: list[Pair] = []
    used_en: set[Path] = set()
    used_es: set[Path] = set()
    used_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Pair CSV has no header.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Pair CSV missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):

            pair_id = row["pair_id"].strip()
            en_name = row["en_file"].strip()
            es_name = row["es_file"].strip()

            if not pair_id or not en_name or not es_name:
                raise ValueError(
                    f"CSV row {row_number}: incomplete pair definition"
                )

            # Basic language sanity check
            if not en_name.upper().startswith("EN_"):
                raise ValueError(
                    f"CSV row {row_number}: expected English file, "
                    f"got {en_name}"
                )

            if not es_name.upper().startswith("ES_"):
                raise ValueError(
                    f"CSV row {row_number}: expected Spanish file, "
                    f"got {es_name}"
                )

            en_path = (audio_dir / en_name).resolve()
            es_path = (audio_dir / es_name).resolve()

            if not en_path.is_file():
                raise FileNotFoundError(
                    f"English file does not exist: {en_path}"
                )

            if not es_path.is_file():
                raise FileNotFoundError(
                    f"Spanish file does not exist: {es_path}"
                )

            if pair_id in used_ids:
                raise ValueError(
                    f"Duplicate pair_id in CSV: {pair_id}"
                )

            if en_path in used_en:
                raise ValueError(
                    f"English file used more than once: {en_name}"
                )

            if es_path in used_es:
                raise ValueError(
                    f"Spanish file used more than once: {es_name}"
                )

            used_ids.add(pair_id)
            used_en.add(en_path)
            used_es.add(es_path)

            pairs.append(
                Pair(
                    pair_id=pair_id,
                    en_path=en_path,
                    es_path=es_path,
                )
            )

    if not pairs:
        raise ValueError("No verified pairs found.")

    return pairs


def build_pair(
    pairs: list[Pair],
    target_seconds: float,
    silence_seconds: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:

    silence = np.zeros(
        int(TARGET_SR * silence_seconds),
        dtype=np.int16,
    )

    en_chunks: list[np.ndarray] = []
    es_chunks: list[np.ndarray] = []

    manifest: list[dict] = []

    en_total = 0.0
    es_total = 0.0

    for pair in pairs:

        en_audio, en_sr = read_wav(pair.en_path)
        es_audio, es_sr = read_wav(pair.es_path)

        if en_sr != TARGET_SR:
            raise ValueError(
                f"{pair.en_path.name}: expected {TARGET_SR} Hz, "
                f"got {en_sr}"
            )

        if es_sr != TARGET_SR:
            raise ValueError(
                f"{pair.es_path.name}: expected {TARGET_SR} Hz, "
                f"got {es_sr}"
            )

        en_dur = duration(en_audio, en_sr)
        es_dur = duration(es_audio, es_sr)

        # Add the ENTIRE bilingual pair or neither side.
        en_chunks.append(en_audio)
        es_chunks.append(es_audio)

        en_total += en_dur
        es_total += es_dur

        manifest.append(
            {
                "pair_id": pair.pair_id,
                "english_file": pair.en_path.name,
                "spanish_file": pair.es_path.name,
                "english_duration": round(en_dur, 3),
                "spanish_duration": round(es_dur, 3),
            }
        )

        # Stop after BOTH recordings reach target duration or all pairs used
        if (
            target_seconds > 0
            and en_total >= target_seconds
            and es_total >= target_seconds
        ):
            break

        en_chunks.append(silence)
        es_chunks.append(silence)

        en_total += silence_seconds
        es_total += silence_seconds

    if not manifest:
        raise RuntimeError("No audio was selected.")

    # Remove trailing silence if present
    if len(en_chunks) > 1 and np.array_equal(en_chunks[-1], silence):
        en_chunks.pop()
        es_chunks.pop()

    return (
        np.concatenate(en_chunks),
        np.concatenate(es_chunks),
        manifest,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build matched English/Spanish evaluation pair from DRAL dataset."
    )

    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
        help="Directory containing DRAL WAV fragments",
    )

    parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="CSV containing explicitly verified EN/ES pairs",
    )

    parser.add_argument(
        "--output-dir",
        default=Path("dral_2min_pair"),
        type=Path,
        help="Output directory for generated WAV pairs and manifest",
    )

    parser.add_argument(
        "--target-seconds",
        default=120.0,
        type=float,
        help="Target length in seconds (0 for all available pairs)",
    )

    parser.add_argument(
        "--silence-seconds",
        default=0.35,
        type=float,
        help="Silence inserted between utterances",
    )

    args = parser.parse_args()

    pairs = load_pairs(
        args.pairs.resolve(),
        args.audio_dir.resolve(),
    )

    print(f"Verified pairs available: {len(pairs)}")

    en_audio, es_audio, manifest = build_pair(
        pairs,
        target_seconds=args.target_seconds,
        silence_seconds=args.silence_seconds,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    en_output = output_dir / "DRAL_EN_2min.wav"
    es_output = output_dir / "DRAL_ES_2min.wav"

    write_wav(en_output, en_audio, TARGET_SR)
    write_wav(es_output, es_audio, TARGET_SR)

    en_duration = duration(en_audio, TARGET_SR)
    es_duration = duration(es_audio, TARGET_SR)

    metadata = {
        "sample_rate": TARGET_SR,
        "target_seconds": args.target_seconds,
        "silence_seconds": args.silence_seconds,
        "pairs_used": len(manifest),
        "english_duration_seconds": round(en_duration, 3),
        "spanish_duration_seconds": round(es_duration, 3),
        "pairs": manifest,
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("DONE")
    print(f"Pairs used:       {len(manifest)}")
    print(f"English duration: {en_duration:.2f}s")
    print(f"Spanish duration: {es_duration:.2f}s")
    print()
    print(f"English:  {en_output}")
    print(f"Spanish:  {es_output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

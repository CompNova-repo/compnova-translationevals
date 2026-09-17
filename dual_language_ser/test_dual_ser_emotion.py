"""Compare English/Spanish emotion with one shared emotion2vec+ model.

Examples:
    python test_dual_ser_emotion.py --self-test
    python test_dual_ser_emotion.py --num-pairs 5
    python test_dual_ser_emotion.py --en-file EN.wav --es-file ES.wav
    python test_dual_ser_emotion.py --device cuda --num-pairs 10

This script evaluates already-translated audio.  It does not translate audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from emotion_comparison import (
    EMOTIONS,
    aggregate,
    compare_pair,
    make_emotion_result,
)
from ser_models import SER_MODEL_HUB, SER_MODEL_NAME, load_shared_ser


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "dral"
DEFAULT_OUT_DIR = ROOT / "outputs" / "ser_single_model"
DEFAULT_NUM_PAIRS = 5


def discover_pairs(data_dir: Path, num_pairs: int) -> list[tuple[Path, Path]]:
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be greater than zero")
    english_files = sorted(data_dir.glob("EN_*.wav"))
    pairs: list[tuple[Path, Path]] = []
    for english_path in english_files:
        suffix = english_path.name[len("EN_") :]
        spanish_path = data_dir / f"ES_{suffix}"
        if spanish_path.is_file():
            pairs.append((english_path, spanish_path))
        if len(pairs) == num_pairs:
            break
    if not pairs:
        raise FileNotFoundError(
            f"no matching EN_*.wav / ES_*.wav pairs found in {data_dir}"
        )
    return pairs


def _resolve_input_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.en_file or args.es_file:
        if not (args.en_file and args.es_file):
            raise ValueError("--en-file and --es-file must be provided together")
        english_path = Path(args.en_file).expanduser().resolve()
        spanish_path = Path(args.es_file).expanduser().resolve()
        if not english_path.is_file():
            raise FileNotFoundError(f"English audio not found: {english_path}")
        if not spanish_path.is_file():
            raise FileNotFoundError(f"Spanish audio not found: {spanish_path}")
        return [(english_path, spanish_path)]
    return discover_pairs(
        Path(args.data_dir).expanduser().resolve(), args.num_pairs
    )


def _print_distribution(title: str, result) -> None:
    print(title)
    for emotion in EMOTIONS:
        print(f"  {emotion:<11} {result.scores[emotion]:.4f}")
    print(f"  top: {result.top_label} ({result.top_score:.4f})")


def _print_pair_report(pair, pair_index: int, total: int) -> None:
    print(f"\nPAIR {pair_index}/{total}")
    print(f"  {Path(pair.source_path).name} <-> {Path(pair.target_path).name}")
    print(f"  shared model: {pair.source.model}")
    _print_distribution("\nEnglish", pair.source)
    _print_distribution("\nSpanish", pair.target)
    print("\nComparison")
    print(f"  top emotion agrees       : {pair.top_label_agrees}")
    print(f"  cosine similarity        : {pair.cosine_similarity:.4f}")
    print(f"  Jensen-Shannon similarity: {pair.js_similarity:.4f}")
    print(f"  mean absolute difference : {pair.mean_absolute_difference:.4f}")
    print(
        "  review reasons           : "
        + (", ".join(pair.review_reasons) if pair.review_reasons else "none")
    )


def _json_pair(pair) -> dict:
    return asdict(pair)


def _csv_row(pair_index: int, pair) -> dict:
    row = {
        "pair_index": pair_index,
        "source_path": pair.source_path,
        "target_path": pair.target_path,
        "model": pair.source.model,
        "source_top_label": pair.source.top_label,
        "source_top_score": pair.source.top_score,
        "target_top_label": pair.target.top_label,
        "target_top_score": pair.target.top_score,
        "top_label_agrees": pair.top_label_agrees,
        "cosine_similarity": pair.cosine_similarity,
        "js_similarity": pair.js_similarity,
        "mean_absolute_difference": pair.mean_absolute_difference,
        "source_non_specific_mass": pair.source.non_specific_mass,
        "target_non_specific_mass": pair.target.non_specific_mass,
        "review_reasons": "|".join(pair.review_reasons),
    }
    for emotion in EMOTIONS:
        row[f"source_{emotion}"] = pair.source.scores[emotion]
        row[f"target_{emotion}"] = pair.target.scores[emotion]
        row[f"absdiff_{emotion}"] = pair.per_emotion_abs_diff[emotion]
    return row


def save_results(pairs, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "single_model_ser_results.json"
    csv_path = out_dir / "single_model_ser_summary.csv"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": SER_MODEL_NAME,
        "model_hub": SER_MODEL_HUB,
        "languages": ["en", "es"],
        "taxonomy": list(EMOTIONS),
        "comparison_design": "same_checkpoint_same_classification_head",
        "calibration_note": (
            "Scores are directly comparable outputs of one model head, but "
            "are not claimed to be human-calibrated probabilities or proof "
            "of equal error rates across languages."
        ),
        "pairs": [_json_pair(pair) for pair in pairs],
        "aggregate": aggregate(pairs),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    rows = [_csv_row(index, pair) for index, pair in enumerate(pairs, 1)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _resolve_device(device: str) -> str:
    if device in {"cpu", "cuda"}:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_self_test() -> None:
    """Fast offline tests; no model download is performed."""
    from ser_models import parse_emotion2vec_output

    raw = [{
        "labels": [
            "生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy",
            "中立/neutral", "其他/other", "难过/sad", "吃惊/surprised",
            "<unk>",
        ],
        "scores": [0.04, 0.03, 0.02, 0.60, 0.20, 0.03, 0.03, 0.04, 0.01],
    }]
    parsed = parse_emotion2vec_output(raw)
    assert tuple(parsed) == EMOTIONS
    assert abs(sum(parsed.values()) - 1.0) < 1e-9
    en = make_emotion_result(parsed, "en", SER_MODEL_NAME)
    es = make_emotion_result(parsed, "es", SER_MODEL_NAME)
    comparison = compare_pair(en, es)
    assert comparison.top_label_agrees
    assert np.isclose(comparison.cosine_similarity, 1.0)
    assert np.isclose(comparison.js_similarity, 1.0)
    assert np.isclose(comparison.mean_absolute_difference, 0.0)
    print("self-test passed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare English and Spanish audio with one shared "
            "emotion2vec+ SER checkpoint."
        )
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS)
    parser.add_argument("--en-file", help="English audio for a single pair")
    parser.add_argument("--es-file", help="Spanish audio for a single pair")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run fast parser/comparison tests without downloading the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    try:
        input_pairs = _resolve_input_pairs(args)
        device = _resolve_device(args.device)
        print(f"loading one shared model on {device}: {SER_MODEL_NAME}")
        shared_model = load_shared_ser(device=device)

        comparisons = []
        for index, (english_path, spanish_path) in enumerate(input_pairs, 1):
            english_scores = shared_model.predict_file(english_path)
            spanish_scores = shared_model.predict_file(spanish_path)
            english_result = make_emotion_result(
                english_scores, language="en", model=SER_MODEL_NAME
            )
            spanish_result = make_emotion_result(
                spanish_scores, language="es", model=SER_MODEL_NAME
            )
            comparison = compare_pair(
                english_result,
                spanish_result,
                source_path=str(english_path),
                target_path=str(spanish_path),
            )
            comparisons.append(comparison)
            _print_pair_report(comparison, index, len(input_pairs))

        summary = aggregate(comparisons)
        print("\nAGGREGATE")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not args.no_save:
            json_path, csv_path = save_results(
                comparisons, Path(args.out_dir).expanduser().resolve()
            )
            print(f"\nresults written to:\n  {json_path}\n  {csv_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

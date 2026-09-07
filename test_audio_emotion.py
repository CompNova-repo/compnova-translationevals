"""Smoke / validation test for the Speech Emotion Recognition pipeline.

Standalone — does **not** import ``DemoApp`` (which eagerly loads
SeamlessM4T, Whisper, COMET, MetricX, and NLLB at module import time).
The SER piece is small enough that we re-implement it here with the exact
same contract as ``DemoApp.do_audio_emotion`` /
``DemoApp.compute_affective_match_score`` and exercise it against the
DRAL paired clips in ``data/dral/``.

The published checkpoint (``transformers 4.8.2`` era) saved its
classifier head under the two-layer ``dense`` + ``output`` layout.
Current ``transformers`` uses a single ``classifier`` Linear and would
silently drop the trained head and re-initialise it with random weights
(producing near-uniform outputs). We rebuild the matching head by hand
and patch the head weights in from the saved safetensors.

Usage:
    python test_audio_emotion.py                # uses data/dral/EN_001_#1.wav + ES_001_#1.wav
    python test_audio_emotion.py path/to/file   # runs against one wav
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DRAL_DIR = ROOT / "data" / "dral"

SER_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]
EXPECTED_LABELS = set(SER_LABELS)


def load_audio_emotion_pipeline():
    """Build the wav2vec2 SER feature extractor + patched head model."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_NAME)
    backbone = Wav2Vec2Model.from_pretrained(SER_MODEL_NAME)
    cfg = backbone.config
    dropout_p = getattr(cfg, "final_dropout", 0.1)
    hidden_size = cfg.hidden_size

    # Sequential order matches the loader in DemoApp.load_audio_emotion_pipeline:
    #   0 = dense (Linear + tanh), 1 = tanh (no params), 2 = dropout (no params),
    #   3 = out_proj (Linear).
    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_size, len(SER_LABELS)),
    )

    class _SERModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, input_values, attention_mask=None):
            outputs = self.backbone(input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            # Hidden is at the backbone output time-resolution
            # (post-conv downsampling), so the input-time attention_mask
            # cannot be applied directly. For single-clip inference we
            # mean-pool across the full sequence.
            pooled = hidden.mean(dim=1)
            return self.classifier(pooled)

    model = _SERModel(backbone, classifier)

    weights_path = hf_hub_download(
        repo_id=SER_MODEL_NAME, filename="model.safetensors"
    )
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
    missing, unexpected = classifier.load_state_dict(head_weights, strict=False)
    print(f"head load: {len(head_weights)} tensors; "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    id2label = {i: lbl for i, lbl in enumerate(SER_LABELS)}
    return feature_extractor, model, id2label


def do_audio_emotion(spec, audio_path: str) -> list:
    """Load audio at 16 kHz mono and return sorted SER predictions."""
    feature_extractor, model, id2label = spec
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs)
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    preds = [
        {"label": id2label[i], "score": float(p)}
        for i, p in enumerate(probs)
    ]
    return sorted(preds, key=lambda x: x["score"], reverse=True)


def compute_affective_match_score(src_preds, mt_preds) -> float:
    """Cosine similarity over the canonical 8-class SER label set."""
    if not src_preds or not mt_preds:
        return 0.0
    src_map = {p["label"]: float(p["score"]) for p in src_preds}
    mt_map  = {p["label"]: float(p["score"]) for p in mt_preds}
    p = np.array([src_map.get(lbl, 0.0) for lbl in SER_LABELS])
    q = np.array([mt_map.get(lbl,  0.0) for lbl in SER_LABELS])
    p_norm = float(np.linalg.norm(p))
    q_norm = float(np.linalg.norm(q))
    if p_norm == 0.0 or q_norm == 0.0:
        return 0.0
    return float(np.dot(p, q) / (p_norm * q_norm))


def _check_run(spec, audio_path: Path) -> dict:
    """Run predictions against one file and assert the public contract."""
    print(f"\n=== {audio_path.name} ===")
    sr = librosa.get_samplerate(str(audio_path))
    duration = librosa.get_duration(path=str(audio_path))
    print(
        f"file sr={sr} Hz  duration={duration:.2f}s  "
        f"size={audio_path.stat().st_size} B"
    )

    preds = do_audio_emotion(spec, str(audio_path))
    assert preds, f"no predictions returned for {audio_path}"

    labels_seen = {p["label"] for p in preds}
    missing = EXPECTED_LABELS - labels_seen
    if missing:
        print(f"  !! pipeline did not surface expected labels: {sorted(missing)}")
    else:
        print(f"  OK all {len(EXPECTED_LABELS)} expected labels present")

    score_sum = sum(p["score"] for p in preds)
    print(f"  softmax mass ~ {score_sum:.4f}")

    top = preds[0]
    print(f"  top-1: {top['label']}  ({top['score']:.4f})")
    for p in preds[:3]:
        print(f"    - {p['label']:<10} {p['score']:.4f}")

    assert top["label"] in EXPECTED_LABELS, (
        f"top-1 label {top['label']!r} not in expected label set"
    )
    assert 0.0 <= top["score"] <= 1.0, "top-1 score out of [0,1]"

    return {"top": top["label"], "preds": preds}


def main(argv: list[str]) -> int:
    spec = load_audio_emotion_pipeline()
    proc, model, id2label = spec
    backbone_name = model.backbone.config._name_or_path
    print(f"loaded model: {backbone_name}")
    print(f"id2label = {id2label}")

    if len(argv) > 1:
        for arg in argv[1:]:
            p = Path(arg)
            if not p.exists():
                print(f"!! skipping non-existent file: {p}")
                continue
            _check_run(spec, p)
        return 0

    if not DRAL_DIR.is_dir():
        print(f"!! DRAL directory not found: {DRAL_DIR}")
        return 1

    en_wav = DRAL_DIR / "EN_001_#1.wav"
    es_wav = DRAL_DIR / "ES_001_#1.wav"
    if en_wav.exists():
        _check_run(spec, en_wav)
    if es_wav.exists():
        _check_run(spec, es_wav)
    if en_wav.exists() and es_wav.exists():
        src = do_audio_emotion(spec, str(en_wav))
        mt  = do_audio_emotion(spec, str(es_wav))
        score = compute_affective_match_score(src, mt)
        print(f"\nAffective match (EN -> ES cosine): {score:.4f}")
        assert -1.0 <= score <= 1.0, "cosine similarity out of [-1,1]"

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

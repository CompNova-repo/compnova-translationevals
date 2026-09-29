"""
Shared evaluation engine for CompNova's S2ST translation evaluation.
Extracted from DemoApp.py so the same pipeline is used by both the
Streamlit demo and this API service, with no duplicated logic.

Model loaders use functools.lru_cache instead of st.cache_resource,
since this module has no Streamlit dependency - it can be imported by
a plain script, a FastAPI app, or a Streamlit app equally.
"""
import os

os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

import sys
import json
import subprocess
import functools

import numpy as np
import torch

SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"HF login warning: {e}")

METRICX_REPO_URL = "https://github.com/google-research/metricx.git"
METRICX_PINNED_COMMIT = "fc4978eb064670f7cc33e93ea4f52d38396b8ae6"


def _ensure_metricx_repo():
    if os.path.isdir(os.path.join("metricx", ".git")) or os.path.exists(
        os.path.join("metricx", "metricx24", "predict.py")
    ):
        return
    subprocess.run(
        ["git", "clone", METRICX_REPO_URL, "metricx"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", "metricx", "checkout", METRICX_PINNED_COMMIT],
        check=True, capture_output=True, text=True,
    )


_ensure_metricx_repo()


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
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


def warm_up():
    """Load every model once. Call this at API startup so the FIRST real
    request doesn't pay the cold-load cost (each model can take a while
    to download/load, as seen repeatedly today)."""
    get_whisper()
    get_comet()
    get_align_model()
    get_sentiment_pipeline()


def transcribe_segments(filepath: str):
    whisper_model = get_whisper()
    segments, info = whisper_model.transcribe(
        filepath,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return [seg.text.strip() for seg in segments if seg.text.strip()]


def join_segment_text(segments) -> str:
    return " ".join(segments).strip()


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


def align_segments_by_content(src_segments, mt_segments, min_similarity: float = 0.5, max_merge: int = 2):
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


def cometeval_batch(pairs):
    if not pairs:
        return []
    comet_model = get_comet()
    data = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    use_gpu = 1 if torch.cuda.is_available() else 0
    out = comet_model.predict(data, batch_size=8, gpus=use_gpu)
    return [round(float(s), 4) for s in out.scores]


def metricx_eval_batch(pairs):
    if not pairs:
        return []
    metricx_dir = "metricx" if os.path.exists("metricx") else "."
    results_dir = os.path.join(metricx_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    input_path = os.path.abspath(os.path.join(results_dir, "metricx_api_input.jsonl"))
    output_path = os.path.abspath(os.path.join(results_dir, "metricx_api_output.jsonl"))

    with open(input_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"source": p["src"], "hypothesis": p["mt"], "reference": ""}) + "\n")

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
    subprocess.run(command, check=True, capture_output=True, cwd=metricx_dir)
    with open(output_path, "r", encoding="utf-8") as f:
        return [round(float(json.loads(line).get("prediction", 0.0)), 4) for line in f]


def sentiment_batch(texts):
    if not texts:
        return []
    return get_sentiment_pipeline()(texts)


def evaluate_pair(src_audio_path: str, mt_audio_path: str, min_similarity: float = 0.5, max_merge: int = 2) -> dict:
    """Full evaluation pipeline: transcribe both files, align chunks by
    content, score with COMET-Kiwi + MetricX-24, run 5-class sentiment.
    Returns a JSON-serializable dict, trimmed to the fields that matter
    for presenting results (internal alignment details like merge type
    and raw similarity score are used to build the chunks but left out
    of the response)."""
    src_segments = transcribe_segments(src_audio_path)
    mt_segments = transcribe_segments(mt_audio_path)
    src_text = join_segment_text(src_segments)
    mt_text = join_segment_text(mt_segments)

    pairs, unmatched_src, unmatched_mt = align_segments_by_content(
        src_segments, mt_segments, min_similarity=min_similarity, max_merge=max_merge
    )

    comet_scores = cometeval_batch(pairs)
    metricx_scores = metricx_eval_batch(pairs)
    src_texts = [p["src"] for p in pairs]
    mt_texts = [p["mt"] for p in pairs]
    src_sent = sentiment_batch(src_texts)
    mt_sent = sentiment_batch(mt_texts)

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
        round(100 * sum(sentiment_matches) / len(sentiment_matches), 1) if sentiment_matches else None
    )

    return {
        "transcripts": {"source": src_text, "target": mt_text},
        "summary": {
            "chunks": len(pairs),
            "unmatched": len(unmatched_src) + len(unmatched_mt),
            "comet_kiwi": avg_comet,
            "metricx_24": avg_metricx,
            "sentiment_agreement_pct": sentiment_agreement_pct,
        },
        "chunks": chunks,
        "unmatched": {"source": unmatched_src, "target": unmatched_mt},
    }

import os

os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

import sys
import json
import tempfile
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import librosa
import torch
from gtts import gTTS
import faster_whisper

# Canonical 8-class label set returned by the wav2vec2 SER model. Defined
# here, near the top, because load_audio_emotion_pipeline() references it
# at MODULE LOAD TIME (models are initialized eagerly, not lazily).
SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

# Speech Emotion Recognition is on hold: the current model
# (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) is trained
# on English-only speech, and applying it to non-English target audio
# gives unreliable results (confirmed via cross-lingual SER research -
# same-language accuracy ~0.93 vs cross-language ~0.46 in published
# benchmarks). Flip this back to True once a genuinely multilingual SER
# model is in place. When False, no SER model is loaded, no audio
# emotion inference runs, and the affective-match UI section is hidden.
ENABLE_SER = False

# =========================================================
# 0. STREAMLIT PAGE CONFIG & HF AUTHENTICATION
# =========================================================
st.set_page_config(
    page_title="Audio Translation & Evaluation Hub",
    page_icon="🎙️",
    layout="wide"
)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        st.warning(f"Hugging Face login failed: {e}")

# =========================================================
# 0b. METRICX REPO (research code, not on PyPI)
# =========================================================
# google-research/metricx has no setup.py / pyproject.toml, so it can't be
# pip-installed. We clone it on first launch and run its scripts directly.
METRICX_REPO_URL = "https://github.com/google-research/metricx.git"
METRICX_PINNED_COMMIT = "fc4978eb064670f7cc33e93ea4f52d38396b8ae6"


def _ensure_metricx_repo():
    if os.path.isdir(os.path.join("metricx", ".git")) or os.path.exists(os.path.join("metricx", "metricx24", "predict.py")):
        return
    try:
        st.info("Cloning google-research/metricx (one-time)…")
        subprocess.run(
            ["git", "clone", METRICX_REPO_URL, "metricx"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", "metricx", "checkout", METRICX_PINNED_COMMIT],
            check=True, capture_output=True, text=True,
        )
    except Exception as e:
        st.warning(f"Could not clone metricx repo automatically ({e}). "
                   "Make sure 'metricx/' exists next to DemoApp.py.")


_ensure_metricx_repo()

# =========================================================
# 1. CACHED MODEL LOADERS
# =========================================================

@st.cache_resource(show_spinner="Loading SeamlessM4T Translation Model...")
def load_seamless_model():
    from transformers import AutoProcessor, SeamlessM4TModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
    m4tmodel = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium").to(device)
    return processor, m4tmodel, device

@st.cache_resource(show_spinner="Loading NLLB-200 Text Translation Model...")
def load_nllb_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    
    # We use the 600M distilled version to prevent Out-of-Memory crashes on cloud tiers
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return tokenizer, model

@st.cache_resource(show_spinner="Loading Faster-Whisper Turbo...")
def load_whisper():
    from faster_whisper import WhisperModel
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    return WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)

@st.cache_resource(show_spinner="Loading COMET-Kiwi Model...")
def load_comet():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    return load_from_checkpoint(model_path)

@st.cache_resource(show_spinner="Loading LaBSE alignment model...")
def load_alignment_model():
    """Multilingual sentence embedder used to align source and translation
    chunks by MEANING rather than by list position. LaBSE is trained
    specifically for cross-lingual sentence matching (109 languages)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/LaBSE")

@st.cache_resource(show_spinner="Loading 5-class sentiment model...")
def load_sentiment_model():
    """Text-based 5-class sentiment classifier (Very Negative / Negative /
    Neutral / Positive / Very Positive), run per aligned chunk pair.
    NOTE: this reads the transcript text only, not vocal tone/delivery -
    two sentences that read identically but were SPOKEN with very
    different emotion will get the same label here. That's a known,
    confirmed limitation (see project notes), not a bug."""
    from transformers import pipeline
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

@st.cache_resource(show_spinner="Loading Speech Emotion Recognition Model...")
def load_audio_emotion_pipeline():
    """Load the Wav2Vec2-XLSR SER model that classifies 8 emotions from audio.

    NOTE: not called while ENABLE_SER is False - see the flag near the
    top of this file.
    """
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    backbone = Wav2Vec2Model.from_pretrained(model_name)
    cfg = backbone.config
    dropout_p = getattr(cfg, "final_dropout", 0.1)
    hidden_size = cfg.hidden_size

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
            outputs = self.backbone(
                input_values, attention_mask=attention_mask
            )
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
            return self.classifier(pooled)

    model = _SERModel(backbone, classifier)

    try:
        weights_path = hf_hub_download(
            repo_id=model_name, filename="model.safetensors"
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
        missing, unexpected = classifier.load_state_dict(
            head_weights, strict=False
        )
        if unexpected:
            st.info(
                f"SER: {len(unexpected)} unexpected head keys in "
                f"checkpoint (e.g. {unexpected[0]!r})."
            )
    except Exception as e:
        st.warning(
            f"SER head could not be patched from checkpoint ({e}). "
            "Predictions will be near-uniform."
        )

    model.eval()
    id2label = {i: lbl for i, lbl in enumerate(SER_LABELS)}
    return feature_extractor, model, id2label

# Initialize default models into cache
processor, m4tmodel, translation_device = load_seamless_model()
whisper_model = load_whisper()
comet_model = load_comet()
align_model = load_alignment_model()
sentiment_pipeline = load_sentiment_model()
if ENABLE_SER:
    audio_emotion_proc, audio_emotion_model, audio_emotion_id2label = load_audio_emotion_pipeline()
else:
    audio_emotion_proc, audio_emotion_model, audio_emotion_id2label = None, None, None
nllb_tokenizer, nllb_model = load_nllb_model()

# =========================================================
# 2. CORE PROCESSING FUNCTIONS
# =========================================================

def save_temp_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

# --- Translation Generation Paths ---

def generate_seamless_audio(source_filepath: str, target_lang_code: str) -> str:
    """End-to-End S2ST translation via SeamlessM4T."""
    audio_array, _ = librosa.load(source_filepath, sr=16000)
    audio_inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt").to(translation_device)
    
    with torch.no_grad():
        output_tokens = m4tmodel.generate(
            **audio_inputs,
            tgt_lang=target_lang_code,
            generate_speech=True
        )
    output_audio_array = output_tokens[0].cpu().numpy().squeeze()
    
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_out.name, output_audio_array, m4tmodel.config.sampling_rate)
    return temp_out.name

def generate_nllb_translation(source_text: str, target_lang: str) -> str:
    """Text-to-Text translation via Meta NLLB-200."""
    tokenizer, model = nllb_tokenizer, nllb_model
    
    nllb_lang_map = {
        "spa": "spa_Latn",
        "fra": "fra_Latn",
        "deu": "deu_Latn",
        "cmn": "zho_Hans",
        "ita": "ita_Latn",
        "por": "por_Latn",
        "jpn": "jpn_Jpan"
    }
    target_nllb_code = nllb_lang_map.get(target_lang, "eng_Latn")
    
    inputs = tokenizer(source_text, return_tensors="pt").to(model.device)
    target_lang_id = tokenizer.lang_code_to_id[target_nllb_code]
    
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=512
    )
    
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output_text.strip()

def generate_tts_audio(text: str, lang_code: str) -> str:
    """Synthesizes text to speech using Google TTS."""
    lang_map = {"spa": "es", "fra": "fr", "deu": "de", "cmn": "zh-CN", "ita": "it", "por": "pt", "jpn": "ja"}
    tts_lang = lang_map.get(lang_code, "en")
    
    tts = gTTS(text, lang=tts_lang)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_out.name)
    return temp_out.name

# --- Transcription (whole-clip AND sentence-level, one Whisper pass) ---

def do_transcribe_segments(filepath: str):
    """Transcribes audio ONCE with Faster-Whisper (VAD-filtered) and
    returns the individual sentence-level segments as plain strings.
    Both the whole-clip transcript (via join_segment_text) and the
    chunk-level evaluation below are derived from this single pass, so
    long audio is never transcribed twice."""
    segments, info = whisper_model.transcribe(
        filepath,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return [seg.text.strip() for seg in segments if seg.text.strip()]


def join_segment_text(segments) -> str:
    return " ".join(segments).strip()


def do_transcribe(filepath: str) -> str:
    """Kept for compatibility with any external callers; internally just
    transcribes segments and joins them."""
    return join_segment_text(do_transcribe_segments(filepath))


# --- Content-based, merge-aware chunk alignment ---

def _encode_spans(texts, max_merge: int):
    """Embed every single segment AND every merge of up to max_merge
    consecutive segments. Returns {(start_index, length): embedding}."""
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
    """Align source and translation segments by MEANING using LaBSE
    embeddings plus a monotonic dynamic-programming alignment (same
    family of technique as bitext-alignment tools like Gale-Church /
    Vecalign). Allows 1-1, 1-2, 2-1 and 2-2 merges, since a sentence
    spoken as one unit in the source is often split (or two combined)
    on the translated side.

    Segments that can't find an acceptable match even after trying
    merges are returned separately as "unmatched" rather than
    force-paired or silently dropped.

    Returns (pairs, unmatched_src, unmatched_mt), where each pair is a
    dict: {src, mt, src_count, mt_count, similarity}.
    """
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


# --- Evaluation Metrics ---

def do_cometeval(src_text: str, mt_text: str) -> float:
    data = [{"src": src_text, "mt": mt_text}]
    use_gpu = 1 if torch.cuda.is_available() else 0
    model_output = comet_model.predict(data, batch_size=1, gpus=use_gpu)
    return round(float(model_output.scores[0]), 4)

def do_metricx_eval(src_text: str, mt_text: str) -> float:
    metricx_dir = "metricx" if os.path.exists("metricx") else "."
    results_dir = os.path.join(metricx_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    input_path = os.path.abspath(os.path.join(results_dir, "metricx_input.jsonl"))
    output_path = os.path.abspath(os.path.join(results_dir, "metricx_output.jsonl"))

    with open(input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"source": src_text, "hypothesis": mt_text, "reference": ""}) + "\n")

    command = [
        sys.executable, "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", "google/metricx-24-hybrid-large-v2p6-bfloat16",
        "--max_input_length", "1536",
        "--batch_size", "1",
        "--input_file", input_path,
        "--output_file", output_path,
        "--qe"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, cwd=metricx_dir)
        with open(output_path, "r", encoding="utf-8") as f:
            return round(float(json.loads(f.readline()).get("prediction", 0.0)), 4)
    except Exception as e:
        st.warning(f"MetricX failed: {e}")
        return None


def do_cometeval_batch(pairs):
    """COMET-Kiwi score for every aligned chunk pair in one batched call."""
    if not pairs:
        return []
    data = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    use_gpu = 1 if torch.cuda.is_available() else 0
    model_output = comet_model.predict(data, batch_size=8, gpus=use_gpu)
    return [round(float(s), 4) for s in model_output.scores]


def do_metricx_eval_batch(pairs):
    """MetricX-24 score for every aligned chunk pair in one batched
    subprocess call (loads the model once, scores every chunk)."""
    if not pairs:
        return []
    metricx_dir = "metricx" if os.path.exists("metricx") else "."
    results_dir = os.path.join(metricx_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    input_path = os.path.abspath(os.path.join(results_dir, "metricx_chunk_input.jsonl"))
    output_path = os.path.abspath(os.path.join(results_dir, "metricx_chunk_output.jsonl"))

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
        "--qe"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, cwd=metricx_dir)
        with open(output_path, "r", encoding="utf-8") as f:
            return [round(float(json.loads(line).get("prediction", 0.0)), 4) for line in f]
    except Exception as e:
        st.warning(f"MetricX (chunk-level) failed: {e}")
        return []


def do_sentiment_batch(texts):
    """5-class sentiment (Very Negative..Very Positive) for a list of
    texts, in one batched call. Text-only - see load_sentiment_model()
    docstring for the known tone/delivery limitation."""
    if not texts:
        return []
    return sentiment_pipeline(texts)


def build_metrics_comparison_chart(chunk_pairs, comet_scores, metricx_scores):
    """Line chart of COMET-Kiwi and MetricX-24 across the sequence of
    aligned chunks, both put on the SAME 0-1 scale (1 = good) so they can
    be compared directly: MetricX (0-25, lower=better) is normalised as
    1 - score/25. X-axis is chunk order, not audio timestamps."""
    import plotly.graph_objects as go

    x = list(range(1, len(chunk_pairs) + 1))
    metricx_norm = [
        round(1 - (s / 25.0), 4) if s is not None else None
        for s in metricx_scores
    ]
    hover_texts = [
        f"EN: {p['src'][:70]}<br>ES: {p['mt'][:70]}" for p in chunk_pairs
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=comet_scores, name="COMET-Kiwi",
        mode="lines+markers",
        hovertext=hover_texts, hoverinfo="text+y",
    ))
    fig.add_trace(go.Scatter(
        x=x[:len(metricx_norm)], y=metricx_norm, name="MetricX-24 (normalised: 1 - score/25)",
        mode="lines+markers",
        hovertext=hover_texts[:len(metricx_norm)], hoverinfo="text+y",
    ))
    fig.update_layout(
        xaxis_title="Chunk # (sequence order in the audio)",
        yaxis=dict(title="Quality score (0-1, higher = better)", range=[0, 1]),
        legend=dict(orientation="h", y=1.15),
        height=420,
        margin=dict(t=60),
        hovermode="x unified",
    )
    return fig


def do_audio_emotion(audio_path: str):
    """Run Speech Emotion Recognition directly on an audio waveform.
    NOTE: only meaningful while ENABLE_SER is True."""
    if not audio_path or not os.path.exists(audio_path):
        return []
    try:
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = audio_emotion_proc(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        with torch.no_grad():
            logits = audio_emotion_model(**inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        preds = [
            {"label": audio_emotion_id2label[i], "score": float(p)}
            for i, p in enumerate(probs)
        ]
        return sorted(preds, key=lambda x: x["score"], reverse=True)
    except Exception as e:
        st.warning(
            f"Audio emotion inference failed for {os.path.basename(audio_path)}: {e}"
        )
        return []


def compute_affective_match_score(src_preds, mt_preds) -> float:
    """Cosine similarity of two SER distributions over SER_LABELS."""
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

# =========================================================
# 2.5 INITIALIZE SESSION STATE FOR HISTORY
# =========================================================
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
    
# =========================================================
# 3. STREAMLIT USER INTERFACE
# =========================================================

st.title("🎙️ Speech-to-Speech Translation & Quality Evaluation")
if ENABLE_SER:
    st.markdown("Compare direct vs cascaded translation architectures and evaluate translation quality together with affective / acoustic tone preservation (Speech Emotion Recognition over the audio waveforms).")
else:
    st.markdown("Compare direct vs cascaded translation architectures and evaluate translation quality (semantic accuracy) sentence-by-sentence, plus text-based sentiment. *Affective / acoustic tone evaluation (SER) is temporarily disabled pending a multilingual model fix.*")

# Initialize the Tabs
tab_main, tab_history = st.tabs(["🔍 Translation & Eval", "📜 Run History"])

with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio("Workflow Mode", ["Generate Translation", "Evaluate Existing Audio Pair"])
    
    PRESETS = {
        "Sample 1 (EN_138_#22.wav)": 
            {"src": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\EN_138_#22.wav", 
                "mt": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\ES_138_#22.wav"
            },
        "Sample 2 (EN_138_#10.wav)": 
                    {"src": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\EN_138_#10.wav", 
                        "mt": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\ES_138_#10.wav"
                    },
        "Sample 3 (EN_138_#8.wav)": 
                    {"src": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\EN_138_#8.wav", 
                        "mt": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\ES_138_#8.wav"
                    },
    }
    
    src_audio_path = None
    mt_audio_path = None
    target_lang = "spa"
    translation_system = None

    if mode == "Generate Translation":
        translation_system = st.radio(
            "Translation Architecture",
            ["SeamlessM4T (Direct S2ST)", "Cascaded (Whisper -> NLLB-200-distilled-600M -> TTS)"]
        )
        
        input_source = st.radio("Source Audio Input", ["Upload File", "Preset"])
        if input_source == "Upload File":
            uploaded_src = st.file_uploader("Upload Source Audio", type=["wav", "mp3"], key="s2s_src")
            if uploaded_src:
                src_audio_path = save_temp_file(uploaded_src)
        else:
            preset_name = st.selectbox("Select Preset", list(PRESETS.keys()))
            src_audio_path = PRESETS[preset_name]["src"]

        target_lang = st.selectbox(
            "Target Language",
            options=["spa", "fra", "deu", "cmn", "ita", "por", "jpn"],
            format_func=lambda x: {"spa": "Spanish", "fra": "French", "deu": "German", "cmn": "Mandarin Chinese", "ita": "Italian", "por": "Portuguese", "jpn": "Japanese"}.get(x, x)
        )

    else:
        upload_src = st.file_uploader("Upload Source Audio", type=["wav", "mp3"], key="eval_src")
        upload_mt = st.file_uploader("Upload Target Audio", type=["wav", "mp3"], key="eval_mt")
        if upload_src and upload_mt:
            src_audio_path = save_temp_file(upload_src)
            mt_audio_path = save_temp_file(upload_mt)

    run_btn = st.button("🚀 Run Workflow", type="primary", use_container_width=True)


# ---------------------------------------------------------
# TAB 1: MAIN EXECUTION
# ---------------------------------------------------------
with tab_main:
    if run_btn:
        if not src_audio_path or not os.path.exists(src_audio_path):
            st.error("Please provide a valid source audio file.")
        else:
            with st.spinner("Processing..."):
                
                # --- Step 1: Handle Audio Generation, and transcribe ONCE
                # (as sentence-level segments) for both whole-clip and
                # chunk-level use downstream. ---
                if mode == "Generate Translation":
                    if translation_system == "SeamlessM4T (Direct S2ST)":
                        with st.status("Translating audio with SeamlessM4T...", expanded=True) as status:
                            mt_audio_path = generate_seamless_audio(src_audio_path, target_lang)
                            status.update(label="Translation audio generated!", state="complete")
                            
                        with st.status("Transcribing source and target audio...", expanded=False):
                            src_segments = do_transcribe_segments(src_audio_path)
                            mt_segments = do_transcribe_segments(mt_audio_path)
                            
                    else: # Cascaded Architecture
                        with st.status("Running cascaded translation pipeline...", expanded=True) as status:
                            st.write("1. Transcribing source audio with Whisper...")
                            src_segments = do_transcribe_segments(src_audio_path)
                            src_text_for_mt = join_segment_text(src_segments)
                            
                            st.write("2. Translating text with NLLB-200...")
                            cascaded_mt_text = generate_nllb_translation(src_text_for_mt, target_lang)
                            
                            st.write("3. Synthesizing translated text into audio (TTS)...")
                            mt_audio_path = generate_tts_audio(cascaded_mt_text, target_lang)
                            status.update(label="Cascaded translation generated!", state="complete")

                        with st.status("Transcribing synthesized audio for strict evaluation...", expanded=False):
                            mt_segments = do_transcribe_segments(mt_audio_path)

                else: # Manual Evaluation Mode
                    with st.status("Transcribing source and translated audio...", expanded=False):
                        src_segments = do_transcribe_segments(src_audio_path)
                        mt_segments = do_transcribe_segments(mt_audio_path)

                src_text = join_segment_text(src_segments)
                mt_text = join_segment_text(mt_segments)

                # --- Step 2: Whole-clip quality (+ SER if enabled) ---
                status_label = (
                    "Calculating affective metrics..."
                    if ENABLE_SER else
                    "Preparing evaluation..."
                )
                with st.status(status_label, expanded=False):
                    # Whole-clip COMET/MetricX are NOT computed separately
                    # here - that would mean loading MetricX's 2.46GB model
                    # from a fresh subprocess TWICE per run (once here, once
                    # again for the chunk batch below). The chunk-level
                    # average, computed once, is used as the headline number
                    # instead - it's also a more honest summary of a
                    # multi-sentence clip than one big-blob score anyway
                    # (see project notes on why whole-clip scoring on long
                    # audio produces out-of-domain, less meaningful numbers).
                    if ENABLE_SER:
                        src_emotion = do_audio_emotion(src_audio_path)
                        mt_emotion  = do_audio_emotion(mt_audio_path)
                        affective_match = compute_affective_match_score(src_emotion, mt_emotion)
                    else:
                        src_emotion, mt_emotion, affective_match = [], [], 0.0

                # --- Step 3: Sentence-level alignment, scoring, sentiment ---
                with st.status("Aligning sentence-level chunks by content...", expanded=False):
                    chunk_pairs, unmatched_src, unmatched_mt = align_segments_by_content(
                        src_segments, mt_segments
                    )
                    if chunk_pairs:
                        chunk_comet_scores = do_cometeval_batch(chunk_pairs)
                        chunk_metricx_scores = do_metricx_eval_batch(chunk_pairs)
                        chunk_src_texts = [p["src"] for p in chunk_pairs]
                        chunk_mt_texts = [p["mt"] for p in chunk_pairs]
                        chunk_src_sentiment = do_sentiment_batch(chunk_src_texts)
                        chunk_mt_sentiment = do_sentiment_batch(chunk_mt_texts)
                    else:
                        chunk_comet_scores, chunk_metricx_scores = [], []
                        chunk_src_sentiment, chunk_mt_sentiment = [], []

            st.success("Workflow Complete!")

            # --- Results Display ---
            st.subheader("🎧 Audio Outputs")
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.markdown("**Source Audio**")
                st.audio(src_audio_path)
            with a_col2:
                st.markdown(f"**Translated Audio** {f'({translation_system})' if mode == 'Generate Translation' else ''}")
                st.audio(mt_audio_path)

            st.divider()

            st.subheader("📝 Transcriptions")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.caption("Source Text")
                st.info(src_text if src_text else "*(No speech detected)*")
            with t_col2:
                st.caption("Translated Text")
                st.info(mt_text if mt_text else "*(No speech detected)*")

            st.divider()

            st.divider()

            st.subheader("📊 Translation Quality")
            if not chunk_pairs:
                st.warning("No chunk pairs could be aligned for this clip (too short, or no acceptable semantic match found).")
                comet_score, metricx_score = None, None
            else:
                c_comet_avg = sum(chunk_comet_scores) / len(chunk_comet_scores) if chunk_comet_scores else None
                c_metricx_avg = sum(chunk_metricx_scores) / len(chunk_metricx_scores) if chunk_metricx_scores else None
                # Used as the headline numbers (see note above on why we
                # don't ALSO run a separate whole-clip MetricX pass).
                comet_score = round(c_comet_avg, 4) if c_comet_avg is not None else None
                metricx_score = round(c_metricx_avg, 4) if c_metricx_avg is not None else None

                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Chunks aligned", f"{len(chunk_pairs)}")
                cc2.metric("Avg COMET-Kiwi", f"{c_comet_avg:.3f}" if c_comet_avg is not None else "N/A")
                cc3.metric("Avg MetricX-24", f"{c_metricx_avg:.3f}" if c_metricx_avg is not None else "N/A")

                st.caption(
                    "📈 Sentence-Level Quality Over the Clip — COMET-Kiwi and MetricX-24 "
                    "scored per aligned chunk, both shown on the same 0-1 scale (1 = good). "
                    "MetricX (natively 0-25, lower = better) is normalised here as "
                    "1 - score/25 so the two lines are directly comparable. Hover a point "
                    "to see the sentence pair it corresponds to."
                )
                fig = build_metrics_comparison_chart(chunk_pairs, chunk_comet_scores, chunk_metricx_scores)
                st.plotly_chart(fig, use_container_width=True)

                if unmatched_src or unmatched_mt:
                    with st.expander(
                        f"⚠️ {len(unmatched_src) + len(unmatched_mt)} segment(s) had no acceptable match - review directly"
                    ):
                        st.caption(
                            "These segments could not be aligned even after trying merges, "
                            "and are NOT included in the chart or averages above - they are "
                            "the most likely candidates for a genuine translation omission "
                            "or a transcription error."
                        )
                        if unmatched_src:
                            st.markdown("**Unmatched source segments:**")
                            for s in unmatched_src:
                                st.markdown(f"- {s}")
                        if unmatched_mt:
                            st.markdown("**Unmatched translation segments:**")
                            for m in unmatched_mt:
                                st.markdown(f"- {m}")

                st.divider()

                st.subheader("💬 Sentence-by-Sentence Sentiment (5-class)")
                st.caption(
                    "Very Negative / Negative / Neutral / Positive / Very Positive, per "
                    "aligned chunk. This reads the TRANSCRIPT TEXT only, not vocal tone or "
                    "delivery - two sentences that read identically but were SPOKEN with "
                    "very different emotion will get the same label here. That is a known, "
                    "confirmed limitation, not a bug - see project notes."
                )
                sent_rows = []
                for idx, p in enumerate(chunk_pairs):
                    s = chunk_src_sentiment[idx] if idx < len(chunk_src_sentiment) else {"label": "n/a", "score": 0.0}
                    m = chunk_mt_sentiment[idx] if idx < len(chunk_mt_sentiment) else {"label": "n/a", "score": 0.0}
                    sent_rows.append({
                        "#": idx + 1,
                        "Source": p["src"],
                        "Src Sentiment": s["label"],
                        "Src Conf": round(float(s["score"]), 2),
                        "Translation": p["mt"],
                        "Tgt Sentiment": m["label"],
                        "Tgt Conf": round(float(m["score"]), 2),
                        "Match": "✅" if s["label"] == m["label"] else "❌",
                    })
                st.dataframe(pd.DataFrame(sent_rows), use_container_width=True, hide_index=True)

            if ENABLE_SER:
                st.divider()
                st.subheader("🎭 Affective & Acoustic Tone Alignment")
                ae_col1, ae_col2, ae_col3 = st.columns(3)
                top_src = src_emotion[0] if src_emotion else {"label": "n/a", "score": 0.0}
                top_mt  = mt_emotion[0]  if mt_emotion  else {"label": "n/a", "score": 0.0}
                with ae_col1:
                    st.markdown("**Source Audio Emotion**")
                    st.markdown(f"<div style='font-size:1.35em;font-weight:600'>{top_src['label']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span style='background:#e6f4ff;color:#003a8c;padding:2px 8px;border-radius:8px;font-size:0.85em'>confidence {top_src['score']:.1%}</span>", unsafe_allow_html=True)
                    if len(src_emotion) > 1:
                        runner_ups = ", ".join(f"{p['label']} ({p['score']:.1%})" for p in src_emotion[1:4])
                        st.caption(f"Runner-ups: {runner_ups}")
                with ae_col2:
                    st.markdown("**Target Audio Emotion**")
                    st.markdown(f"<div style='font-size:1.35em;font-weight:600'>{top_mt['label']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span style='background:#fff4e6;color:#7a4a00;padding:2px 8px;border-radius:8px;font-size:0.85em'>confidence {top_mt['score']:.1%}</span>", unsafe_allow_html=True)
                    if len(mt_emotion) > 1:
                        runner_ups = ", ".join(f"{p['label']} ({p['score']:.1%})" for p in mt_emotion[1:4])
                        st.caption(f"Runner-ups: {runner_ups}")
                with ae_col3:
                    st.markdown("**Affective / Acoustic Tone Match**")
                    st.metric("Match %", f"{affective_match:.1%}")
                    match_top1 = top_src["label"] == top_mt["label"] and top_src["label"] != "n/a"
                    st.caption("Top-1 label match: ✅" if match_top1 else "Top-1 label match: ❌")
            else:
                top_src = {"label": "n/a", "score": 0.0}
                top_mt = {"label": "n/a", "score": 0.0}
                st.caption("🎭 Affective / acoustic tone evaluation (SER) is temporarily disabled.")

            record = {
                "Workflow": mode,
                "Architecture": translation_system if mode == "Generate Translation" else "Manual Pair",
                "Translated Lang": target_lang if mode == "Generate Translation" else "N/A",
                "Source Audio": os.path.basename(src_audio_path),
                "Translated Audio": os.path.basename(mt_audio_path),
                "Source Text": src_text,
                "Translated Text": mt_text,
                "COMET-Kiwi": comet_score,
                "MetricX-24": metricx_score,
                "Chunks Aligned": len(chunk_pairs),
                "Chunks Unmatched": len(unmatched_src) + len(unmatched_mt),
                "Avg COMET-Kiwi (chunked)": round(sum(chunk_comet_scores) / len(chunk_comet_scores), 4) if chunk_comet_scores else None,
                "Avg MetricX-24 (chunked)": round(sum(chunk_metricx_scores) / len(chunk_metricx_scores), 4) if chunk_metricx_scores else None,
            }
            if chunk_pairs and chunk_src_sentiment and chunk_mt_sentiment:
                agree = sum(
                    1 for s, m in zip(chunk_src_sentiment, chunk_mt_sentiment) if s["label"] == m["label"]
                ) / len(chunk_pairs)
                record["Sentiment Agreement %"] = round(agree * 100, 1)
            if ENABLE_SER:
                record.update({
                    "Source Audio Emotion": top_src["label"],
                    "Target Audio Emotion": top_mt["label"],
                    "Affective Match Score": round(affective_match, 4),
                })
            st.session_state.eval_history.append(record)

# ---------------------------------------------------------
# TAB 2: RUN HISTORY
# ---------------------------------------------------------
with tab_history:
    st.header("📜 Session Run History")
    
    if not st.session_state.eval_history:
        st.info("No evaluations have been run yet. Process an audio file to see your results accumulate here.")
    else:
        history_df = pd.DataFrame(st.session_state.eval_history)
        st.dataframe(history_df, use_container_width=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full History as CSV",
                data=csv_data,
                file_name="translation_evaluation_history.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with btn_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.eval_history.clear()
                st.rerun()

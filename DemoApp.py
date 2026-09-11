import os
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

# hy mt, requires newer transformers version
# @st.cache_resource(show_spinner="Loading HY-MT1.5-1.8B Text Translation Model...")
# def load_hy_mt_model():
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained("tencent/HY-MT1.5-1.8B")
    
#     # Use bfloat16 to save memory if running on GPU
#     dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
#     model = AutoModelForCausalLM.from_pretrained(
#         "tencent/HY-MT1.5-1.8B",
#         torch_dtype=dtype,
#         device_map="auto" if torch.cuda.is_available() else None
#     )
#     if not torch.cuda.is_available():
#         model = model.to("cpu")
#     return tokenizer, model

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

# slow whisper model
# @st.cache_resource(show_spinner="Loading Whisper ASR Model...")
# def load_whisper():
#     import whisper
#     return whisper.load_model("large")

@st.cache_resource(show_spinner="Loading Faster-Whisper Turbo...")
def load_whisper():
    from faster_whisper import WhisperModel
    import torch
    
    # Automatically use GPU and FP16 (16-bit float) if available for massive speedups
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    # Load the highly optimized 'turbo' model
    return WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)

@st.cache_resource(show_spinner="Loading COMET-Kiwi Model...")
def load_comet():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    return load_from_checkpoint(model_path)

@st.cache_resource(show_spinner="Loading Speech Emotion Recognition Model...")
def load_audio_emotion_pipeline():
    """Load the Wav2Vec2-XLSR SER model that classifies 8 emotions from audio.

    Replaces the legacy text-based ``tabularisai`` classifiers with an audio
    classifier that operates on the raw waveform. We **don't** use the
    high-level ``transformers.pipeline`` because on this checkpoint it
    silently drops the trained classifier head (the published weights were
    saved under the old ``dense`` + ``output`` two-layer head from
    ``transformers 4.8.2``; current ``Wav2Vec2ForSequenceClassification``
    uses a single ``classifier`` Linear and re-initialises the head with
    random weights, producing near-uniform outputs).

    Instead we assemble the backbone via ``Wav2Vec2Model.from_pretrained``
    and rebuild the matching ``dense`` + ``tanh`` + ``dropout`` +
    ``out_proj`` head, then patch the head weights in by hand from the
    saved checkpoint.

    Returns ``(feature_extractor, model, id2label)``; ``do_audio_emotion``
    runs them manually and applies ``softmax`` over the logits so the
    output shape matches what the pipeline would have produced.
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

    # Old-style classifier head: ``dense`` (Linear + tanh) → ``out_proj``
    # (Linear). Implemented as an ``nn.Sequential`` so ``state_dict`` keys
    # line up predictably: index 0 = dense, index 1 = tanh (no params),
    # index 2 = dropout (no params), index 3 = out_proj.
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
            hidden = outputs.last_hidden_state  # (B, T, H)
            # Hidden is at the backbone output time-resolution
            # (post-conv downsampling), so the input-time attention_mask
            # cannot be applied directly. For single-clip inference we
            # mean-pool across the full sequence.
            pooled = hidden.mean(dim=1)
            return self.classifier(pooled)

    model = _SERModel(backbone, classifier)

    # Pull the saved head weights straight from the checkpoint and copy
    # them into our rebuilt head. Keys in the file are
    # ``classifier.dense.*`` and ``classifier.output.*``; we rename
    # ``output`` → index 3 of the Sequential.
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
        # We expect no missing keys (the Sequential has 4 modules, indices
        # 0 and 3 carry weights). Log unexpected for visibility.
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
audio_emotion_proc, audio_emotion_model, audio_emotion_id2label = load_audio_emotion_pipeline()
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
    audio_inputs = processor(audios=audio_array, sampling_rate=16000, return_tensors="pt").to(translation_device)
    
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

# def generate_hy_mt_translation(source_text: str, target_lang: str) -> str:
#     """Text-to-Text translation via Tencent HY-MT1.5-1.8B."""
#     tokenizer, model = load_hy_mt_model()
    
#     # Map 3-letter UI codes to full language names for the Hunyuan prompt
#     lang_names = {
#         "spa": "Spanish", "fra": "French", "deu": "German", 
#         "cmn": "Chinese", "ita": "Italian", "por": "Portuguese", "jpn": "Japanese"
#     }
#     target_name = lang_names.get(target_lang, "English")
    
#     prompt = f"Translate the following segment into {target_name}, without additional explanation.\n{source_text}"
#     messages = [{"role": "user", "content": prompt}]
    
#     tokenized_chat = tokenizer.apply_chat_template(
#         messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)
    
#     outputs = model.generate(
#         tokenized_chat, 
#         max_new_tokens=512,
#         top_k=20, 
#         top_p=0.6, 
#         repetition_penalty=1.05, 
#         temperature=0.7
#     )
    
#     input_length = tokenized_chat.shape[1]
#     output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
#     return output_text.strip()

def generate_nllb_translation(source_text: str, target_lang: str) -> str:
    """Text-to-Text translation via Meta NLLB-200."""
    tokenizer, model = nllb_tokenizer, nllb_model
    
    # NLLB uses specific BCP-47 codes (Language_Script)
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
    
    # 1. Tokenize the input text
    inputs = tokenizer(source_text, return_tensors="pt").to(model.device)
    
    # 2. Grab the specific ID for the target language to force the model to translate to it
    target_lang_id = tokenizer.lang_code_to_id[target_nllb_code]
    
    # 3. Generate the translation
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=512
    )
    
    # 4. Decode the output tokens back into readable text
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

# --- Evaluation Metrics ---

# slow whisper model
# def do_transcribe(filepath: str) -> str:
#     import whisper
#     audio = whisper.load_audio(filepath)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)
#     options = whisper.DecodingOptions()
#     return whisper.decode(whisper_model, mel, options).text

def do_transcribe(filepath: str) -> str:
    """Transcribes audio using optimized Faster-Whisper with VAD preprocessing."""
    
    # We no longer need to manually load, pad, or create mel spectrograms!
    # faster-whisper handles optimal chunking under the hood.
    
    segments, info = whisper_model.transcribe(
        filepath, 
        beam_size=5,
        vad_filter=True, # Preprocessing: Instantly strips out silence!
        vad_parameters=dict(min_silence_duration_ms=500) # Aggressively cuts dead air
    )
    
    # The transcription is a generator, so we iterate through it to get the text
    full_text = " ".join([segment.text for segment in segments])
    
    return full_text.strip()

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

# Canonical 8-class label set returned by the wav2vec2 SER model.
# Kept in this fixed order so cosine similarity is computed against the
# same axis for every pair of source / target predictions.
SER_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]


def do_audio_emotion(audio_path: str):
    """Run Speech Emotion Recognition directly on an audio waveform.

    Loads the file with ``librosa`` at 16 kHz mono (the sampling rate the
    wav2vec2 backbone expects), passes the waveform through the cached
    ``AutoFeatureExtractor`` + custom backbone-plus-head module from
    ``load_audio_emotion_pipeline``, applies ``softmax`` over the logits,
    and returns ``[{"label": str, "score": float}, ...]`` sorted
    high-to-low.
    """
    if not audio_path or not os.path.exists(audio_path):
        return []
    try:
        # librosa.load returns float32 mono audio at the requested rate.
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
    """Cosine similarity of two SER distributions over ``SER_LABELS``.

    An exact top-1 match (with otherwise identical distributions) yields
    ~1.0. Returns 0.0 if either side is missing or degenerate.
    """
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
st.markdown("Compare direct vs cascaded translation architectures and evaluate translation quality together with affective / acoustic tone preservation (Speech Emotion Recognition over the audio waveforms).")

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
                
                # --- Step 1: Handle Audio Generation ---
                if mode == "Generate Translation":
                    if translation_system == "SeamlessM4T (Direct S2ST)":
                        with st.status("Translating audio with SeamlessM4T...", expanded=True) as status:
                            mt_audio_path = generate_seamless_audio(src_audio_path, target_lang)
                            status.update(label="Translation audio generated!", state="complete")
                            
                        with st.status("Transcribing source and target audio...", expanded=False):
                            src_text = do_transcribe(src_audio_path)
                            mt_text = do_transcribe(mt_audio_path)
                            
                    else: # Cascaded Architecture
                        with st.status("Running cascaded translation pipeline...", expanded=True) as status:
                            st.write("1. Transcribing source audio with Whisper...")
                            src_text = do_transcribe(src_audio_path)
                            
                            st.write("2. Translating text with HY-MT1.5-1.8B...")
                            # cascaded_mt_text = generate_hy_mt_translation(src_text, target_lang)
                            cascaded_mt_text = generate_nllb_translation(src_text, target_lang)
                            
                            st.write("3. Synthesizing translated text into audio (TTS)...")
                            mt_audio_path = generate_tts_audio(cascaded_mt_text, target_lang)
                            status.update(label="Cascaded translation generated!", state="complete")

                        with st.status("Transcribing synthesized audio for strict evaluation...", expanded=False):
                            # We transcribe the TTS audio so the evaluation strictly scores the final audio output
                            mt_text = do_transcribe(mt_audio_path)

                else: # Manual Evaluation Mode
                    with st.status("Transcribing source and translated audio...", expanded=False):
                        src_text = do_transcribe(src_audio_path)
                        mt_text = do_transcribe(mt_audio_path)

                # --- Step 2: Quality & Paralinguistic Evaluation ---
                with st.status(
                    "Calculating quality and affective (audio emotion) metrics...",
                    expanded=False,
                ):
                    comet_score = do_cometeval(src_text, mt_text)
                    metricx_score = do_metricx_eval(src_text, mt_text)
                    src_emotion = do_audio_emotion(src_audio_path)
                    mt_emotion  = do_audio_emotion(mt_audio_path)
                    affective_match = compute_affective_match_score(src_emotion, mt_emotion)

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

            st.subheader("📊 Translation Quality Estimation")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("COMET-Kiwi Score", f"{comet_score}")
            m_col2.metric("MetricX-24 QE Score", f"{metricx_score}" if metricx_score is not None else "N/A")

            st.subheader("🎭 Affective & Acoustic Tone Alignment")
            ae_col1, ae_col2, ae_col3 = st.columns(3)
            top_src = src_emotion[0] if src_emotion else {"label": "n/a", "score": 0.0}
            top_mt  = mt_emotion[0]  if mt_emotion  else {"label": "n/a", "score": 0.0}
            with ae_col1:
                st.markdown("**Source Audio Emotion**")
                st.markdown(
                    f"<div style='font-size:1.35em;font-weight:600'>{top_src['label']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span style='background:#e6f4ff;color:#003a8c;padding:2px 8px;"
                    f"border-radius:8px;font-size:0.85em'>confidence {top_src['score']:.1%}</span>",
                    unsafe_allow_html=True,
                )
                if len(src_emotion) > 1:
                    runner_ups = ", ".join(
                        f"{p['label']} ({p['score']:.1%})"
                        for p in src_emotion[1:4]
                    )
                    st.caption(f"Runner-ups: {runner_ups}")
            with ae_col2:
                st.markdown("**Target Audio Emotion**")
                st.markdown(
                    f"<div style='font-size:1.35em;font-weight:600'>{top_mt['label']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span style='background:#fff4e6;color:#7a4a00;padding:2px 8px;"
                    f"border-radius:8px;font-size:0.85em'>confidence {top_mt['score']:.1%}</span>",
                    unsafe_allow_html=True,
                )
                if len(mt_emotion) > 1:
                    runner_ups = ", ".join(
                        f"{p['label']} ({p['score']:.1%})"
                        for p in mt_emotion[1:4]
                    )
                    st.caption(f"Runner-ups: {runner_ups}")
            with ae_col3:
                st.markdown("**Affective / Acoustic Tone Match**")
                st.metric("Match %", f"{affective_match:.1%}")
                match_top1 = (
                    top_src["label"] == top_mt["label"]
                    and top_src["label"] != "n/a"
                )
                st.caption(
                    "Top-1 label match: ✅"
                    if match_top1 else "Top-1 label match: ❌"
                )

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
                "Source Audio Emotion": top_src["label"],
                "Target Audio Emotion": top_mt["label"],
                "Affective Match Score": round(affective_match, 4),
            }
            st.session_state.eval_history.append(record)

# ---------------------------------------------------------
# TAB 2: RUN HISTORY
# ---------------------------------------------------------
with tab_history:
    st.header("📜 Session Run History")
    
    if not st.session_state.eval_history:
        st.info("No evaluations have been run yet. Process an audio file to see your results accumulate here.")
    else:
        # Convert session state list to DataFrame
        history_df = pd.DataFrame(st.session_state.eval_history)
        
        # Display the interactive dataframe
        st.dataframe(history_df, use_container_width=True)
        
        # Export and Clear Buttons
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
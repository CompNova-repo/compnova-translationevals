import os
import tempfile
import json
import subprocess
import sys
import pandas as pd
import streamlit as st

# =========================================================
# 0. SECURITY & CONFIGURATION
# =========================================================
st.set_page_config(page_title="Audio Translation Eval", layout="wide", page_icon="🎙️")

# TIP: Replace hardcoded tokens with an environment variable or st.secrets
# os.environ.get("HF_TOKEN", "your_token_here")
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", "YOUR_HF_TOKEN")
except:
    try:
        HF_TOKEN = os.environ.get("HF_TOKEN")
    except:
        st.warning("HF token not found.")

from huggingface_hub import login
if HF_TOKEN != "YOUR_HF_TOKEN":
    login(token=HF_TOKEN)

    
# =========================================================
# 1. CACHED MODEL LOADERS
# =========================================================
# Caching ensures models are loaded into memory only ONCE on startup.

@st.cache_resource(show_spinner="Loading Whisper Large...")
def load_whisper():
    import whisper
    return whisper.load_model("large")

@st.cache_resource(show_spinner="Loading COMET Model...")
def load_comet():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    return load_from_checkpoint(model_path)

@st.cache_resource(show_spinner="Loading Sentiment & Emotion Models...")
def load_nlp_pipelines():
    from transformers import pipeline, AutoTokenizer
    
    senti_pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=False)
    emot_pipe = pipeline(
        "text-classification",
        model="tabularisai/multilingual-emotion-classification",
        tokenizer=tokenizer,
        function_to_apply="sigmoid",
        top_k=None,
    )
    return senti_pipe, emot_pipe

# Initialize models
whisper_model = load_whisper()
comet_model = load_comet()
sentiment_pipeline, emotion_pipeline = load_nlp_pipelines()

# =========================================================
# 2. EVALUATION FUNCTIONS
# =========================================================

def do_transcribe(filepath: str) -> str:
    import whisper
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)
    
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result.text

def do_cometeval(src_text: str, mt_text: str) -> float:
    data = [{"src": src_text, "mt": mt_text}]
    model_output = comet_model.predict(data, batch_size=1, gpus=0 if str(whisper_model.device) == "cpu" else 1)
    return round(float(model_output.scores[0]), 4)

def do_metricx_eval(src_text: str, mt_text: str) -> float:
    os.makedirs("results", exist_ok=True)
    input_path = os.path.abspath("results/metricx_input.jsonl")
    output_path = os.path.abspath("results/metricx_output.jsonl")

    metricx_item = {"source": src_text, "hypothesis": mt_text, "reference": ""}
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metricx_item) + "\n")

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
        subprocess.run(command, check=True, capture_output=True)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
            return round(float(data.get("prediction", 0.0)), 4)
    except Exception as e:
        st.warning(f"MetricX could not run: {e}")
        return None

def do_sentiment(src_text: str, mt_text: str):
    return sentiment_pipeline([src_text, mt_text])

def do_emotion(src_text: str, mt_text: str):
    return emotion_pipeline([src_text, mt_text])

def save_temp_file(uploaded_file) -> str:
    """Helper to save uploaded Streamlit file buffer to disk for Whisper."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

# =========================================================
# 3. STREAMLIT USER INTERFACE
# =========================================================

st.title("🎙️ Audio Translation Evaluation Hub")
st.markdown("Compare source audio against translated output using state-of-the-art Quality Estimation (QE), Sentiment, and Emotion metrics.")

# Define your preset catalog
PRESETS = {
    "Conversation Sample 1 (EN -> ES)": {
        "src": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\EN_138_#10.wav",
        "mt": r"C:\Users\erich\Documents\Compnova\topi-full-data\DRAL testset data\fragments-long\fragments-long-MM\ES_138_#10.wav",
        "description": "English conversation segment translated to Spanish."
    }
}

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    input_mode = st.radio("Input Source", ["Presets", "Upload Custom Audio"])
    src_audio_path, mt_audio_path = None, None

    if input_mode == "Presets":
        preset_choice = st.selectbox("Choose a Preset Pair:", list(PRESETS.keys()))
        selected = PRESETS[preset_choice]
        src_audio_path = selected["src"]
        mt_audio_path = selected["mt"]
        st.caption(selected["description"])
    else:
        uploaded_src = st.file_uploader("Upload Source Audio", type=["wav", "mp3", "m4a"], key="src_upload")
        uploaded_mt = st.file_uploader("Upload Translated/Target Audio", type=["wav", "mp3", "m4a"], key="mt_upload")
        
        if uploaded_src and uploaded_mt:
            src_audio_path = save_temp_file(uploaded_src)
            mt_audio_path = save_temp_file(uploaded_mt)

    translation_model = st.selectbox(
        "Translation System Evaluated",
        ["SeamlessM4T", "Whisper + NMT", "AudioPaLM", "Custom Model"]
    )
    
    run_btn = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

# --- Main Display & Audio Previews ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Audio (Original)")
    if src_audio_path and os.path.exists(src_audio_path):
        st.audio(src_audio_path)
    else:
        st.info("No source audio loaded.")

with col2:
    st.subheader("Target Audio (Translated)")
    if mt_audio_path and os.path.exists(mt_audio_path):
        st.audio(mt_audio_path)
    else:
        st.info("No translated audio loaded.")

st.divider()

# --- Execution & Results ---
if run_btn:
    if not (src_audio_path and mt_audio_path and os.path.exists(src_audio_path) and os.path.exists(mt_audio_path)):
        st.error("Please provide valid source and translated audio files before running evaluation.")
    else:
        with st.spinner("Transcribing audio and running multi-metric evaluation..."):
            # 1. Transcriptions
            src_text = do_transcribe(src_audio_path)
            mt_text = do_transcribe(mt_audio_path)
            
            # 2. Evaluations
            comet_score = do_cometeval(src_text, mt_text)
            metricx_score = do_metricx_eval(src_text, mt_text)
            senti_results = do_sentiment(src_text, mt_text)
            emot_results = do_emotion(src_text, mt_text)

        st.success("Evaluation Complete!")

        # --- Section A: Transcriptions ---
        st.markdown("### 📝 Transcriptions")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.caption("**Source Text**")
            st.write(f"> *{src_text}*")
        with t_col2:
            st.caption(f"**Translated Text ({translation_model})**")
            st.write(f"> *{mt_text}*")

        # --- Section B: Core Metric Scorecards ---
        st.markdown("### 📊 Translation Quality Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric("COMET-Kiwi Score", f"{comet_score}", help="Higher is better (Reference-less quality estimation).")
        with m_col2:
            metricx_display = f"{metricx_score}" if metricx_score is not None else "N/A"
            st.metric("MetricX-24 Score", metricx_display, help="Lower scores typically denote higher similarity/quality depending on QE configuration.")
        with m_col3:
            st.metric("Model Evaluated", translation_model)

        # --- Section C: Paralinguistic Checks (Sentiment & Emotion) ---
        st.markdown("### 🎭 Sentiment & Emotion Alignment")
        s_col1, s_col2 = st.columns(2)
        
        with s_col1:
            st.write("**Source Tone:**")
            st.markdown(f"- **Sentiment:** `{senti_results[0]['label']}` ({senti_results[0]['score']:.2%})")
            top_src_emotions = sorted(emot_results[0], key=lambda x: x['score'], reverse=True)[:3]
            st.markdown("- **Top Emotions:** " + ", ".join([f"{e['label']} ({e['score']:.1%})" for e in top_src_emotions]))
            
        with s_col2:
            st.write("**Translated Tone:**")
            st.markdown(f"- **Sentiment:** `{senti_results[1]['label']}` ({senti_results[1]['score']:.2%})")
            top_mt_emotions = sorted(emot_results[1], key=lambda x: x['score'], reverse=True)[:3]
            st.markdown("- **Top Emotions:** " + ", ".join([f"{e['label']} ({e['score']:.1%})" for e in top_mt_emotions]))

        # --- Section D: Exportable DataFrame ---
        with st.expander("📁 View / Download Raw Results"):
            summary_df = pd.DataFrame([{
                "Model": translation_model,
                "Source Text": src_text,
                "Target Text": mt_text,
                "COMET Score": comet_score,
                "MetricX Score": metricx_score,
                "Source Sentiment": senti_results[0]['label'],
                "Target Sentiment": senti_results[1]['label']
            }])
            st.dataframe(summary_df, use_container_width=True)
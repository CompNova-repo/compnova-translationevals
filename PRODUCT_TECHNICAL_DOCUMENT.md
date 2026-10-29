# CompNova Translation Evaluation Hub — Technical Product Document

> **Audience:** Technical Product Managers, Engineering Leads, Stakeholders  
> **Purpose:** End-to-end understanding of the system — what it does, how it works, key design decisions, and operational realities.  
> **Scope:** Covers the Streamlit demo (`DemoApp.py`), the SER CLI (`run_ser_eval.py`), test harness, data tooling, and the model ecosystem. Not a line-by-line code review.

---

## 1. Executive Summary

**What is this?**  
A **speech-to-speech translation (S2ST) evaluation platform** packaged as a Streamlit web app + CLI tools. It lets you:

1. **Translate spoken audio** between 7 languages using two fundamentally different architectures:
   - **Direct S2ST** — One model (`facebook/hf-seamless-m4t-medium`) takes raw waveform → translated waveform.
   - **Cascaded** — `faster-whisper` (ASR) → `NLLB-200` (text MT) → `gTTS` (TTS).
2. **Evaluate the output** with **reference-free quality metrics** (no human reference needed):
   - **COMET-Kiwi** — semantic adequacy/fluency, 0–1 (higher = better).
   - **MetricX-24 QE** — error-density proxy, 0–25 (lower = better).
3. **Compare paralinguistic tone** (emotion/prosody) between source and translated audio via a **Speech Emotion Recognition (SER)** audio classifier — *currently disabled pending a multilingual model*.

**Why does it exist?**  
To put the two S2ST architectures side-by-side and surface *what each one preserves, mistranslates, or flattens* — not just at the word level but at the acoustic/tone level. The SER component was added because text-based sentiment/emotion classifiers conflate emotion with transcription accuracy and tokenizer coverage; the audio classifier evaluates the actual signal the listener hears.

**Current state:**  
- Core translation + text-quality pipeline: **functional** (runs on CPU or GPU).  
- SER (audio emotion): **disabled by default** (`ENABLE_SER = False`) — the available checkpoint is English-only and cross-lingual accuracy is ~0.46 vs ~0.93 same-language (per published benchmarks).  
- Deployed on Hugging Face Spaces (Streamlit SDK); `README.md` front-matter points to `app.py` (expected to mirror `DemoApp.py`).

---

## 2. User Flows (What the Human Sees)

### 2.1 Streamlit App (`DemoApp.py`)

**Sidebar → Mode Selector**
| Mode | What Happens |
|------|--------------|
| **Generate Translation** | User picks architecture (SeamlessM4T or Cascaded), uploads/chooses source audio, picks target language → app generates translated audio, transcribes both sides, runs all evaluators, shows results. |
| **Evaluate Existing Audio Pair** | User uploads *both* source and target audio → skips generation, runs transcription + evaluation only. |

**Tabs**
| Tab | Purpose |
|-----|---------|
| **Translation & Eval** | Main workspace: audio players, transcriptions, quality scores (COMET/MetricX), sentence-level charts, sentiment table, SER panel (when enabled). |
| **Run History** | Accumulates every run in `st.session_state.eval_history`; downloadable as CSV. Not persisted across browser sessions. |

**Key UI Elements**
- **Preset samples** — 3 hard-coded ToPi/DRAL clips (Windows paths baked in; only work on the original dev machine).
- **Language selector** — 7 targets: Spanish, French, German, Mandarin, Italian, Portuguese, Japanese (UI shows names, internal codes are `spa`, `fra`, `deu`, `cmn`, `ita`, `por`, `jpn`).
- **Progress status widgets** — `st.status` blocks for each pipeline stage (translation, transcription, evaluation).
- **Visualisations** — Plotly line chart comparing COMET & normalised MetricX across aligned sentence chunks.

### 2.2 CLI: `run_ser_eval.py`

Standalone SER evaluation (no Streamlit, no heavy translation models). Two modes:

```bash
# Single pair
python run_ser_eval.py --source data/dral/EN_001_#1.wav --target data/dral/ES_001_#1.wav

# Batch over a directory (auto-pairs EN_* ↔ ES_* by base name)
python run_ser_eval.py --batch-dir data/dral --pattern "EN_001_#*.wav"
```

Outputs:
- Console: per-clip top-1 emotion + confidence, full 8-class distribution, pairwise cosine match (ANSI-coloured if TTY).
- Markdown log: appended to `SER_EVAL_LOG.md` (or custom `--md-out`) with timestamped tables.

### 2.3 Smoke Test: `test_audio_emotion.py`

Validates the SER pipeline contract:
1. Model + custom head load correctly.
2. Inference returns all 8 expected labels.
3. Top-1 label ∈ expected set, score ∈ [0,1].
4. Cosine similarity between paired EN/ES clips is finite.

Run: `python test_audio_emotion.py` (defaults to `data/dral/EN_001_#1.wav` + `ES_001_#1.wav`).

---

## 3. System Architecture — Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (DemoApp.py)                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Sidebar      │  │ Tab: Translation │  │ Tab: Run History     │  │
│  │ Mode/Lang/   │──►│ & Eval           │──►│ (session_state      │  │
│  │ Audio Input  │  │ Audio/Transcript/│  │  eval_history)       │  │
│  └──────────────┘  │ Metrics/Chart    │  │ CSV Download         │  │
│                    └──────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌───────────────┐   ┌─────────────┐
   │ Direct S2ST │    │ Cascaded      │   │ Manual Pair │
   │ SeamlessM4T │    │ Whisper→NLLB→ │   │ (no gen)    │
   └──────┬──────┘    │ gTTS          │   └──────┬──────┘
          │           └───────┬───────┘          │
          ▼                   ▼                   ▼
   ┌─────────────────────────────────────────────────────────┐
   │              TRANSCRIPTION (both sides)                 │
   │        faster-whisper large-v3-turbo (VAD-filtered)     │
   │  • Whole-clip text  • Sentence-level segments (once)    │
   └─────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌───────────────┐   ┌─────────────┐
   │ COMET-Kiwi  │    │ MetricX-24 QE │   │ SER (audio) │
   │ (Python)    │    │ (subprocess)  │   │ wav2vec2    │
   └──────┬──────┘    └───────┬───────┘   └──────┬──────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │ DISPLAY + HISTORY   │
                   │ Audio players       │
                   │ Transcripts         │
                   │ Scores + Chart      │
                   │ Sentiment table     │
                   │ SER panel (opt)     │
                   │ CSV export          │
                   └─────────────────────┘
```

---

## 4. Translation Architectures — Deep Dive

### 4.1 Direct S2ST — SeamlessM4T (facebook/hf-seamless-m4t-medium)

| Aspect | Detail |
|--------|--------|
| **Model** | `facebook/hf-seamless-m4t-medium` — unified speech-to-speech & speech-to-text model. |
| **Loader** | `AutoProcessor` + `SeamlessM4TModel.from_pretrained`, cached via `@st.cache_resource`. |
| **Device** | CUDA + `bfloat16` if GPU, else CPU + `float32`. |
| **Inference** | `processor(audio, sampling_rate=16000)` → `model.generate(..., tgt_lang=code, generate_speech=True)` inside `torch.no_grad()`. |
| **Output** | Waveform at model's native sampling rate → written to temp `.wav` via `soundfile.write`. |
| **Latency** | Single forward pass; no intermediate text exposed (but we re-transcribe for evaluation). |
| **Languages** | Target language code passed directly to `generate(tgt_lang=...)`. |

**Why this architecture?**  
End-to-end differentiable; preserves prosody/acoustics better in theory; fewer error-compounding stages.

### 4.2 Cascaded — Whisper → NLLB-200 → gTTS

| Stage | Model / Tool | Key Config |
|-------|--------------|------------|
| **ASR** | `faster-whisper` `large-v3-turbo` | `beam_size=5`, `vad_filter=True`, `min_silence_duration_ms=500` |
| **MT** | `facebook/nllb-200-distilled-600M` | `forced_bos_token_id` for target lang; 600M variant fits HF Spaces free tier |
| **TTS** | `gTTS` (Google Text-to-Speech) | Maps UI code → BCP-47: `spa→es`, `fra→fr`, `deu→de`, `cmn→zh-CN`, `ita→it`, `por→pt`, `jpn→ja` |

**Critical detail:** The cascaded path **re-transcribes the synthesized MP3** before scoring. This means quality metrics evaluate *what the user actually hears* (TTS artifacts included), not the intermediate NLLB text.

**Why this architecture?**  
Modular; each component can be swapped/improved independently; NLLB covers 200+ languages; gTTS is lightweight and zero-dependency.

### 4.3 Manual Pair Mode

No generation. User supplies both source and target audio. Same transcription + evaluation pipeline. Useful for:
- Evaluating third-party S2ST systems.
- Regression testing with known-good/bad pairs.
- Human-in-the-loop studies.

---

## 5. Evaluation Layer — The Metrics

### 5.1 Text-Based Quality (Reference-Free)

| Metric | Model | Input | Range | Direction | Integration |
|--------|-------|-------|-------|-----------|-------------|
| **COMET-Kiwi** | `Unbabel/wmt22-cometkiwi-da` | `(src_text, mt_text)` | 0–1 | Higher = better | Native Python (`comet` pkg) |
| **MetricX-24 QE** | `google/metricx-24-hybrid-large-v2p6-bfloat16` | `(src_text, mt_text)` | 0–25 | Lower = better | Subprocess (`metricx24.predict`) |

**Why reference-free?**  
No human reference translations needed — works on *any* source/target pair.

**MetricX subprocess design:**  
MetricX research code has no `setup.py`/`pyproject.toml`; cannot be `pip install`ed. The app clones the repo (`git clone https://github.com/google-research/metricx.git` at a pinned commit) on first launch, then shells out to `python -m metricx24.predict --qe ...` for each batch. Input/output via JSONL files in `metricx/results/`.

**Whole-clip vs Chunk-level:**  
The app **does not** run a separate whole-clip MetricX/COMET pass. Instead, it:
1. Splits both transcriptions into sentence-level segments (VAD-filtered Whisper).
2. Aligns source ↔ target segments by *meaning* (LaBSE embeddings + DP alignment, allowing 1-1, 1-2, 2-1, 2-2 merges).
3. Scores each aligned chunk pair in batch.
4. Reports **average of chunk scores** as the headline number.

*Rationale:* Long audio → single huge text blob → out-of-domain for QE models (trained on sentence pairs). Chunk-level is more honest and granular.

### 5.2 Sentence-Level Alignment (Content-Based)

**Problem:** Source and translation often have different sentence boundaries (one spoken sentence → two written, or vice versa). Naïve positional alignment fails.

**Solution:** `align_segments_by_content()` — embeddings + monotonic DP alignment.

1. **Encode all spans** — For each side, embed every single segment *and* every merge of up to `max_merge=2` consecutive segments using **LaBSE** (`sentence-transformers/LaBSE`, 109 languages, cross-lingual sentence matching).
2. **Similarity matrix** — Cosine similarity between source spans and target spans.
3. **Dynamic programming** — Monotonic path through the grid (like Gale-Church/Vecalign), maximizing sum of similarities, with a `min_similarity=0.5` threshold.
4. **Output** — Aligned pairs (with merge counts + similarity), plus unmatched source/target segments surfaced separately for review.

### 5.3 Paralinguistic / Affective Evaluation (SER) — **Currently Disabled**

| Component | Detail |
|-----------|--------|
| **Model** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |
| **Backbone** | Wav2Vec2-Large-XLSR (self-supervised on multilingual speech). |
| **Head** | Custom 2-layer classifier (Linear → Tanh → Dropout → Linear) patched from safetensors checkpoint (original `dense` + `output` layout ≠ current `transformers` `classifier` layout). |
| **Input** | Raw 16 kHz mono waveform (via `librosa.load(sr=16000, mono=True)`). |
| **Labels** | 8 classes: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`. |
| **Output** | Sorted `[{label, score}]` per clip (softmax probabilities). |
| **Match Score** | Cosine similarity between the two 8-d probability vectors. 1.0 = identical distributions; 0 = orthogonal. |
| **Why audio, not text?** | Text classifiers (legacy `tabularisai` models) depend on Whisper transcripts → conflate emotion with ASR errors & target-language tokenizer gaps. Audio SER evaluates the *acoustic signal the listener hears*. |

**Disable Flag:** `ENABLE_SER = False` at top of `DemoApp.py`. When `False`:
- SER model not loaded (saves ~300 MB RAM + load time).
- No audio emotion inference runs.
- SER UI section hidden; caption shows "temporarily disabled pending a multilingual model fix."

**Re-enabling:** Flip `ENABLE_SER = True` once a genuinely multilingual SER checkpoint exists (e.g., XLSR fine-tuned on multilingual emotion data).

### 5.4 Legacy Text-Based Sentiment (Still Active)

| Model | `tabularisai/multilingual-sentiment-analysis` |
|-------|-----------------------------------------------|
| Type | 5-class: Very Negative / Negative / Neutral / Positive / Very Positive |
| Input | Transcript text only (per aligned chunk). |
| Limitation | **Ignores vocal tone/delivery** — two sentences reading identically but spoken with different emotion get the same label. Documented as known limitation, not a bug. |
| Use | Chunk-level sentiment agreement % logged in history. |

---

## 6. Data Flow — End-to-End Walkthrough

```
User Action: "Run Workflow" (Generate Translation → SeamlessM4T → Spanish)
────────────────────────────────────────────────────────────────────────
1. AUDIO INGEST
   - Uploaded file → tempfile (preserves extension) OR preset path.
   - Source audio path secured.

2. TRANSLATION GENERATION
   - generate_seamless_audio(src_path, "spa")
     • librosa.load(src_path, sr=16000) → 16 kHz mono array
     • processor → tensors on device
     • m4tmodel.generate(tgt_lang="spa", generate_speech=True)
     • output waveform → temp .wav file → mt_audio_path

3. TRANSCRIPTION (BOTH SIDES, ONCE EACH)
   - do_transcribe_segments(src_audio_path) → list of sentence strings
   - do_transcribe_segments(mt_audio_path) → list of sentence strings
   - VAD-filtered, beam_size=5, single pass per file.
   - Whole-clip text = " ".join(segments) for display.

4. WHOLE-CLIP SER (if ENABLE_SER)
   - do_audio_emotion(src_audio_path) → 8-class distribution
   - do_audio_emotion(mt_audio_path) → 8-class distribution
   - compute_affective_match_score() → cosine similarity

5. SENTENCE-LEVEL ALIGNMENT & SCORING
   - align_segments_by_content(src_segments, mt_segments)
     → chunk_pairs, unmatched_src, unmatched_mt
   - If pairs exist:
     • do_cometeval_batch(chunk_pairs) → list of COMET scores
     • do_metricx_eval_batch(chunk_pairs) → list of MetricX scores
     • do_sentiment_batch(src_texts) + do_sentiment_batch(mt_texts)

6. DISPLAY
   - Audio players (source + translated)
   - Transcriptions side-by-side
   - Headline metrics: chunks aligned, avg COMET, avg MetricX
   - Plotly chart: COMET + normalised MetricX (1 - score/25) per chunk
   - Unmatched segments expander (if any)
   - Sentiment table per chunk (source vs target, match ✅/❌)
   - SER panel (if enabled): top-1 emotions + affective match %

7. HISTORY
   - Record dict appended to st.session_state.eval_history
   - Fields: mode, architecture, lang, file names, texts, scores, chunk counts,
             sentiment agreement %, SER fields (if enabled).
   - Tab 2: DataFrame view + CSV download button + Clear History.
```

---

## 7. Model Inventory & Resource Profile

| Model / Component | Size / Weight | Load Method | Device Strategy | Notes |
|-------------------|---------------|-------------|-----------------|-------|
| SeamlessM4T (S2ST) | ~1.5 GB | `@st.cache_resource` | CUDA `bfloat16` / CPU `float32` | Processor + model |
| faster-whisper | ~1.5 GB | `@st.cache_resource` | CUDA `float16` / CPU `int8` | `large-v3-turbo` |
| NLLB-200 600M | ~1.2 GB | `@st.cache_resource` | CUDA `bfloat16` / CPU `float32` | Distilled for memory |
| COMET-Kiwi | ~400 MB | `@st.cache_resource` | CUDA / CPU (internal) | `unbabel-comet` pkg |
| LaBSE | ~470 MB | `@st.cache_resource` | CPU (SentenceTransformer) | Alignment only |
| Sentiment (text) | ~270 MB | `@st.cache_resource` | CPU | `tabularisai/multilingual-sentiment-analysis` |
| SER (audio) | ~300 MB | `@st.cache_resource` | CUDA / CPU | **Disabled by default** |
| MetricX-24 | ~2.5 GB | **Subprocess** (not loaded in Python) | Spawned process uses GPU if available | Cloned from git; runs `--qe` mode |
| gTTS | negligible | On-demand | N/A | Cloud TTS (needs internet) |

**Total resident memory (SER disabled, GPU):** ~5–6 GB model weights + overhead.  
**Total resident memory (SER enabled):** +~300 MB.

**HF Spaces Free Tier Fit:**  
- NLLB 600M (not 1.3B/3.3B) chosen explicitly to fit 16 GB RAM.  
- `packages.txt` provides `ffmpeg` for audio decoding.  
- `requirements.txt` pins `git+https://github.com/google-research/metricx.git` for reproducible MetricX clone.

---

## 8. Configuration & Secrets

| Setting | Source | Required For |
|---------|--------|--------------|
| `HF_TOKEN` | `os.environ["HF_TOKEN"]` → `st.secrets["HF_TOKEN"]` | Gated models: COMET-Kiwi, MetricX-24, XLM-R tokenizer (NLLB). Template in `example_secrets.toml`. |
| Language codes | Hard-coded maps in `DemoApp.py` | UI (`spa`/`fra`/…) → NLLB (`spa_Latn`/…) → gTTS (`es`/`fr`/…). |
| `ENABLE_SER` | Module-level constant in `DemoApp.py` | Flip to `True` when multilingual SER model available. |
| `METRICX_PINNED_COMMIT` | `fc4978eb064670f7cc33e93ea4f52d38396b8ae6` | Reproducible MetricX clone. |

**Never commit a real `HF_TOKEN`.** Use `.streamlit/secrets.toml` on HF Spaces or env var locally.

---

## 9. Operational Realities & Known Constraints

### 9.1 Preset Paths Are Machine-Specific
```python
PRESETS = {
    "Sample 1": {"src": r"C:\Users\erich\Documents\Compnova\topi-full-data\...", ...}
}
```
These **only work on the original dev Windows machine**. On HF Spaces / other environments, users must upload files. *Fix:* Ship sample files in repo or download from HF dataset at runtime.

### 9.2 MetricX Subprocess Overhead
- Each `do_metricx_eval_batch` call spawns a fresh Python process, loads the 2.5 GB model, scores, exits.
- First run clones the repo (~30s). Subsequent runs still pay model-load latency per batch.
- **Optimization opportunity:** Keep a persistent MetricX server (FastAPI) or use `transformers`-native port if available.

### 9.3 SER Cross-Lingual Gap
- Current checkpoint: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (English-only fine-tune).
- Published same-language accuracy ~0.93; cross-language ~0.46.
- **Do not enable for production multilingual use** — results will be misleading.

### 9.4 Text Sentiment ≠ Vocal Emotion
- `tabularisai/multilingual-sentiment-analysis` reads *transcript text*.
- "I'm fine" spoken angrily → "Neutral" label.
- This is a **known, documented limitation** — not a bug. SER was added to address it.

### 9.5 Session History Is Ephemeral
- `st.session_state.eval_history` lives only in the browser session.
- No server-side persistence. CSV download is the only export.
- For multi-user / persistent history, need a database (SQLite, Postgres, etc.).

### 9.6 No Authentication / Multi-Tenancy
- Single-user Streamlit app. No login, no quotas, no isolation.
- HF Spaces deployment is public by default.

### 9.7 Error Handling Is UI-Centric
- `st.warning()` / `st.error()` for user-visible issues.
- MetricX failures return `None` / empty list; pipeline continues.
- No structured logging / observability (no Prometheus, no Sentry).

---

## 10. Development & Testing Workflow

### 10.1 Local Setup
```bash
# 1. Python deps (MetricX cloned from git)
pip install -r requirements.txt

# 2. System dep
# Ubuntu/Debian: apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: choco install ffmpeg OR manual PATH

# 3. Auth (for gated models)
export HF_TOKEN="hf_..."  # or create .streamlit/secrets.toml

# 4. Optional: fetch DRAL subset (30 EN/ES pairs → data/dral/)
python download_dral_subset.py

# 5. Run app
streamlit run DemoApp.py
```

### 10.2 Running Tests
```bash
# SER smoke test (validates loader, 8 labels, cosine contract)
python test_audio_emotion.py

# SER CLI single pair
python run_ser_eval.py --source data/dral/EN_001_#1.wav --target data/dral/ES_001_#1.wav

# SER CLI batch
python run_ser_eval.py --batch-dir data/dral --pattern "EN_001_#*.wav"
```

### 10.3 Notebooks (Documentation / Exploration)
| Notebook | Purpose |
|----------|---------|
| `SoftwareDemo1.ipynb` | Narrative walk-through of legacy building blocks (Whisper → COMET → MetricX → text sentiment/emotion). Kept for historical context. |
| `TestOutputs.ipynb` | Early worked examples over ToPi corpus; builds `process_audio_pipeline` helper that outputs a blended accuracy DataFrame. Superseded by SER columns. |

---

## 11. Extensibility & Future Work (Product View)

### 11.1 High-Impact Near-Term
| Item | Why | Effort |
|------|-----|--------|
| **Multilingual SER model** | Unblocks the flagship affective-match feature. | Research / fine-tune XLSR on multilingual emotion data. |
| **Persistent history DB** | Enables longitudinal studies, multi-user, audit trail. | Add SQLite/Postgres + auth. |
| **MetricX as a service** | Eliminates per-batch subprocess cold-start (10–30s). | Wrap in FastAPI / Triton / modal.com. |
| **Sample audio bundled / auto-fetched** | Presets work everywhere; better first-run UX. | Download ToPi/DRAL samples at build or runtime. |
| **Batch evaluation UI** | Current UI is single-pair; researchers need 100s. | Add "Upload CSV of pairs" → background job → results table. |

### 11.2 Architectural Improvements
| Area | Current | Target |
|------|---------|--------|
| **Model loading** | Eager at import (all 6+ models) | Lazy / on-demand per mode |
| **Alignment** | LaBSE + DP (max_merge=2) | Vecalign-style with audio timestamps |
| **Evaluation** | Chunk-average headline | Configurable aggregation (mean, min, harmonic) |
| **Metrics** | COMET/MetricX only | Add xCOMET, GEMBA, TASER, BLEURT as toggles |
| **TTS** | gTTS (cloud, limited voices) | Local TTS (XTTS, VITS, Piper) for offline + voice control |

### 11.3 Productisation Checklist
- [ ] Config file (YAML/TOML) for model versions, thresholds, language maps.
- [ ] Structured logging (JSON) + log aggregation.
- [ ] Health checks / readiness probes for container deployment.
- [ ] CI/CD: lint, type-check, smoke test on PR.
- [ ] Automated benchmark regression (fixed dataset → metric deltas).
- [ ] Accessibility audit (WCAG) for Streamlit UI.

---

## 12. File Reference Cheat Sheet

| File | Role | Key Exports / Entry Points |
|------|------|----------------------------|
| `DemoApp.py` | **Main Streamlit app** | `load_seamless_model`, `load_nllb_model`, `load_whisper`, `load_comet`, `load_alignment_model`, `load_sentiment_model`, `load_audio_emotion_pipeline`, `generate_seamless_audio`, `generate_nllb_translation`, `generate_tts_audio`, `do_transcribe_segments`, `align_segments_by_content`, `do_cometeval*`, `do_metricx_eval*`, `do_audio_emotion`, `compute_affective_match_score`, UI tabs |
| `run_ser_eval.py` | **SER CLI** | `main()`, `_predict_all`, `_pair_from_predictions`, `_append_markdown` |
| `test_audio_emotion.py` | **SER smoke test** | `load_audio_emotion_pipeline`, `do_audio_emotion`, `compute_affective_match_score`, `main()` |
| `download_dral_subset.py` | **Data fetcher** | Downloads 30 EN/ES pairs from `jonavila/DRAL` → `data/dral/` |
| `ARCHITECTURE.md` | **Architecture spec** | Component map, data flow, per-layer write-up |
| `README.md` | **HF Spaces front-matter + quickstart** | `app_file: app.py` (must mirror `DemoApp.py`) |
| `SER_EVAL_LOG.md` | **Accumulated SER runs** | Markdown log appended by `run_ser_eval.py` |
| `requirements.txt` | **Python deps** | Includes `git+https://github.com/google-research/metricx.git` |
| `packages.txt` | **System deps (HF Spaces)** | `ffmpeg` |
| `example_secrets.toml` | **Secrets template** | `HF_TOKEN` placeholder |

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **S2ST** | Speech-to-Speech Translation (audio → audio). |
| **Cascaded** | Multi-stage pipeline: ASR → MT → TTS. |
| **Direct / End-to-End** | Single model: audio → audio. |
| **QE (Quality Estimation)** | Reference-free MT evaluation (no human reference needed). |
| **COMET-Kiwi** | Unbabel's reference-free QE model (regression score 0–1). |
| **MetricX-24** | Google's reference-free QE (error-density score 0–25). |
| **SER** | Speech Emotion Recognition — audio classification of emotion/prosody. |
| **Affective Match** | Cosine similarity between source & target SER probability vectors. |
| **LaBSE** | Language-agnostic BERT Sentence Embedding (109 languages). |
| **VAD** | Voice Activity Detection (filters silence). |
| **DP Alignment** | Dynamic Programming alignment (monotonic sequence matching). |
| **HF Spaces** | Hugging Face's free hosting for ML demos (Streamlit, Gradio, etc.). |

---

## 14. Quick Answers to Common Questions

**Q: "Can I run this on CPU-only?"**  
Yes. All models fall back to CPU (`int8` for Whisper, `float32` for others). Expect 5–10× slower generation; evaluation still works.

**Q: "Why does MetricX take so long on first run?"**  
It clones the Google Research repo (~30s) and downloads the 2.5 GB model weights. Subsequent runs only pay model-load time per batch.

**Q: "Can I add a new target language?"**  
Add the 3-letter code to the `st.selectbox` options, then extend `nllb_lang_map` and `lang_map` (gTTS) in `DemoApp.py`. NLLB-200 supports 200+ languages; gTTS supports ~100.

**Q: "How do I evaluate a custom S2ST model?"**  
Use **Manual Pair mode** — upload your source + model's output audio. Same metrics, no generation code changes.

**Q: "Why is SER disabled?"**  
The only readily-available wav2vec2 SER checkpoint is English-only. Cross-lingual accuracy is unacceptably low (~0.46). Enabling it would produce misleading "affective match" numbers for non-English targets.

**Q: "Can I replace gTTS with a local TTS?"**  
Yes — swap `generate_tts_audio()` to call XTTS, VITS, Piper, etc. Keep the same signature (text + lang_code → temp audio path).

**Q: "Is the chunk alignment perfect?"**  
No. `max_merge=2` and `min_similarity=0.5` are heuristics. Unmatched segments are surfaced for manual review. For production, consider timestamp-aware alignment (Vecalign, WhisperX).

---

## 15. Appendix: SER Log Sample (from `SER_EVAL_LOG.md`)

```
## Run @ 2026-09-07 14:00:25 UTC
**Mode:** `batch`  **Files evaluated:** 32  **Pairs:** 16

### Pairwise cosine match
| Source | Target | Source top-1 | Target top-1 | Match |
| --- | --- | --- | --- | --- |
| EN_001_#1.wav | ES_001_#1.wav | surprised (0.9031) | surprised (0.9383) | **0.9997** |
| EN_001_#8.wav | ES_001_#8.wav | surprised (0.8514) | sad (0.6470) | **0.0230** |
| EN_001_#15.wav | ES_001_#15.wav | calm (0.5017) | sad (0.8987) | **0.3969** |
```

*Interpretation:* High match (~0.99) = translated audio preserves the source's emotional tone. Low match (~0.02) = tone shifted dramatically (e.g., surprised → sad).

---

*Document generated from codebase analysis — no code was modified. For implementation details, see `ARCHITECTURE.md` and inline docstrings in `DemoApp.py`.*
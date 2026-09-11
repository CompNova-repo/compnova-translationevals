---
title: S2ST Eval Demo
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Translation & Audio-Evaluation Hub

A Streamlit demo that **translates spoken audio** between seven languages
and **evaluates** the output with reference-free quality metrics plus a
**Speech Emotion Recognition (SER)** audio classifier that compares the
paralinguistic tone of the source and target clips.

The two contrastive translation pipelines are:

- **Direct S2ST** — `facebook/hf-seamless-m4t-medium` consumes the source
  waveform and emits a translated waveform in one shot.
- **Cascaded** — `faster-whisper` (`large-v3-turbo`) transcribes the
  source, `facebook/nllb-200-distilled-600M` translates the text, and
  `gTTS` synthesizes the target-language audio.

Evaluators:

- **COMET-Kiwi** (`Unbabel/wmt22-cometkiwi-da`) — 0–1, higher is better.
- **MetricX-24 QE** (`google/metricx-24-hybrid-large-v2p6-bfloat16`) —
  0–25, lower is better.
- **Speech Emotion Recognition** — the `wav2vec2-lg-xlsr-en-speech
  -emotion-recognition` audio classifier (see *Affective tone* below).

The previous text-based `tabularisai/multilingual-sentiment-analysis`
and `tabularisai/multilingual-emotion-classification` models are gone;
the SER audio classifier reads tone directly off the waveform so the
paralinguistic evaluation is independent of transcription accuracy and
target-language tokenizer coverage.

## Repository layout

| Path | Role |
| --- | --- |
| `DemoApp.py` | The Streamlit app — page config, model loaders, processing functions, sidebar, two tabs (Translation & Eval, Run History). Entry point referenced by the HF Spaces frontmatter (`app_file: app.py`; the deployed copy should mirror `DemoApp.py`). |
| `run_ser_eval.py` | **CLI** driver for the SER pipeline — run on a single pair or batch over a directory, prints to console and appends a timestamped markdown log. |
| `test_audio_emotion.py` | Smoke / regression test for `run_ser_eval.py` and `DemoApp.do_audio_emotion`. Asserts the loader, all 8 expected labels, and the cosine-similarity contract. |
| `SoftwareDemo1.ipynb` | Narrative walk-through of the legacy building blocks (transcription, COMET, MetricX). Kept for historical context. |
| `TestOutputs.ipynb` | Early worked examples over the [ToPi](https://www.nigelward.com/topi-full-data.zip) corpus. |
| `ARCHITECTURE.md` | Component map, end-to-end data flow, and detailed write-up of every layer (translation architectures, evaluation layer, SER engine). |
| `download_dral_subset.py` | Helper that pulls a 30-pair EN/ES subset of the [DRAL](https://huggingface.co/datasets/jonavila/DRAL) dataset into `data/dral/`. |
| `SER_EVAL_LOG.md` | Output of `run_ser_eval.py` — accumulates one timestamped section per run. |
| `requirements.txt` | Python dependencies. |
| `packages.txt` | `ffmpeg` — required system dep for audio decoding on HF Spaces. |
| `example_secrets.toml` | Template for `HF_TOKEN`. Don't commit a real token. |
| `LICENSE` | Repository license. |

## Affective tone (SER)

The audio emotion classifier runs on the **raw 16 kHz mono waveform** of
both source and target clips and emits a probability distribution over
eight emotions:

```
['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
```

An **Affective Match** score is computed as the cosine similarity between
the two 8-dimensional probability vectors. Identical distributions yield
1.0; identical top-1 labels with the rest of the mass concentrated on the
same runner-up land close to 1.0; orthogonal distributions land near 0.

> **Why not text?** The legacy `tabularisai` classifiers relied on
> `faster-whisper` transcriptions, so they conflated emotion with
> transcription accuracy. The wav2vec2-XLSR SER model evaluates
> paralinguistic tone directly from the acoustic signal — the property
> the listener actually hears.

## Quick start

```bash
# 1. Install Python deps. MetricX is pulled straight from the
#    google-research repo so HF Spaces ships reproducibly.
pip install -r requirements.txt

# 2. (One time) fetch a 30-pair EN/ES subset of the DRAL dataset.
python download_dral_subset.py            # writes data/dral/EN_*.wav + ES_*.wav

# 3. (Optional) authenticate so gated models download correctly.
echo 'HF_TOKEN = "hf_..."' > .streamlit/secrets.toml
# or: export HF_TOKEN="hf_..."

# 4. Run the Streamlit UI.
streamlit run DemoApp.py
```

`ffmpeg` must be installed (managed by `packages.txt` on HF Spaces).

## CLI: SER evaluation

For non-UI runs, `run_ser_eval.py` exposes the same SER pipeline.

```bash
# Single pair — cosine match between two clips
python run_ser_eval.py \
    --source data/dral/EN_001_#1.wav \
    --target data/dral/ES_001_#1.wav

# Batch over a glob — also auto-pairs clips that share a base name
# minus their language prefix (e.g. EN_001_#1.wav ↔ ES_001_#1.wav)
python run_ser_eval.py --batch-dir data/dral --pattern "EN_001_#*.wav"

# Append to a custom markdown log
python run_ser_eval.py --batch-dir data/dral \
    --pattern "EN_*.wav" --md-out reports/morning.md

# No console output, markdown only
python run_ser_eval.py --batch-dir data/dral --quiet
```

Both modes print per-clip top-1 labels + confidences to the console
(ANSI-coloured when stdout is a tty, honouring `NO_COLOR`) and append a
timestamped section to `SER_EVAL_LOG.md` containing per-clip tables and
pairwise cosine matches.

## Smoke test

```bash
python test_audio_emotion.py [optional_wav.wav ...]
```

Asserts that:

1. The wav2vec2 SER backbone + custom-built classifier head load.
2. Inference returns a list of `{label, score}` dicts spanning **all 8**
   expected emotions.
3. Top-1 label is in the expected set and score ∈ [0, 1].
4. Cosine similarity between paired EN/ES clips is finite.

When no argument is given the test runs against
`data/dral/EN_001_#1.wav` and `data/dral/ES_001_#1.wav` from the DRAL
subset.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the component map,
end-to-end data-flow diagram, and the per-layer write-up (translation
architectures, evaluation layer, and the new SER engine).

## Configuration

- **HF_TOKEN** — read from `os.environ` first, then `st.secrets["HF_TOKEN"]`.
  Required only for gated model downloads (COMET-Kiwi, MetricX-24). The
  `example_secrets.toml` is a template; never commit a real token.
- **Language codes** — the UI shows three-letter codes (`spa`, `fra`,
  `deu`, `cmn`, `ita`, `por`, `jpn`); each maps independently to the
  format expected by NLLB (`spa_Latn`, etc.) and gTTS (`es`, `fr`, …).
- **Eval history** — lives in `st.session_state.eval_history`; not
  persisted across browser sessions. The "Download Full History as CSV"
  button in tab 2 is the only way to keep a record.

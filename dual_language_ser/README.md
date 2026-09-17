# English–Spanish SER with one shared model

This replaces the previous two-model design with one loaded
`iic/emotion2vec_plus_large` checkpoint. Both English and Spanish files pass
through the same weights, classifier head, label order, and score scale.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell (use this instead of the line above)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements_ser.txt
python test_dual_ser_emotion.py --self-test
```

The first real run downloads the model weights (approximately 2 GB), so it
requires an internet connection and free disk space.

## Run one pair

```bash
python test_dual_ser_emotion.py \
  --en-file path/to/english.wav \
  --es-file path/to/spanish.wav
```

## Run a DRAL-style folder

Place matched files under `data/dral/`:

```text
data/dral/EN_001.wav
data/dral/ES_001.wav
```

Then run:

```bash
python test_dual_ser_emotion.py --num-pairs 5
```

Use `--device cuda` for an NVIDIA GPU or omit it for automatic selection.
JSON and CSV results are written under `outputs/ser_single_model/`.

## Run sentence-by-sentence SER

`sentence_ser_eval.py` uses word-level Whisper timestamps to construct
sentence spans, aligns English and Spanish spans in order with multilingual
LaBSE embeddings, and runs the same emotion2vec checkpoint on both audio
slices. It supports 1:1, 1:2, 2:1, and 2:2 sentence alignment.

Run the model-free verification first:

```bash
python sentence_ser_eval.py --self-test
```

Then evaluate a bilingual recording pair:

```bash
python sentence_ser_eval.py \
  --source-file path/to/english.wav \
  --target-file path/to/spanish.wav \
  --source-language en \
  --target-language es
```

The default multilingual Whisper model is `small`. For higher ASR accuracy on
a suitable GPU, pass `--whisper-model large-v3 --device cuda`. The script
loads ASR, alignment, and SER models in sequence to reduce peak memory use.

Outputs are written to `outputs/sentence_ser/`:

- `sentence_ser_results.json` contains ASR sentences, unmatched indices,
  alignments, complete emotion distributions, review flags, and session
  aggregates.
- `sentence_ser_results.csv` is a flat per-aligned-span export.

Session cosine, Jensen-Shannon similarity, and top-label agreement are reported
both as simple means and weighted by the average source/target span duration.
Sentence boundaries come from Whisper's inferred punctuation, with configurable
duration and word-count fallbacks for unpunctuated speech.

## Interpretation boundary

One shared checkpoint fixes the cross-model calibration mismatch. It does not
prove that the model has identical accuracy or probability calibration for
English and Spanish. Treat cosine/JS values as candidate preservation metrics
until checked against bilingual human labels; do not publish them as an
absolute emotion-accuracy percentage.

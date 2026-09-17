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

## Interpretation boundary

One shared checkpoint fixes the cross-model calibration mismatch. It does not
prove that the model has identical accuracy or probability calibration for
English and Spanish. Treat cosine/JS values as candidate preservation metrics
until checked against bilingual human labels; do not publish them as an
absolute emotion-accuracy percentage.

---
title: S2ST Eval Demo
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: DemoApp.py
pinned: false
---

# CompNova — Speech-to-Speech Translation Evaluation

A working demo of CompNova's evaluation engine for spoken translation
(S2ST). Hand it a clip in the source language and the clip that was
produced by a translation system, and it reports — sentence pair by
sentence pair — how well the translation held up.

Two services make up the demo:

| Service | Port | What it is |
| --- | --- | --- |
| Evaluation API | `8000` | REST API, with interactive docs at `/docs` |
| Dashboard | `8501` | Streamlit monitor of every stored evaluation |

> **A note on the models.** The scoring and alignment models behind the
> engine are deliberately **not named** — not in the dashboard, not in the
> API responses, not in the OpenAPI page, and not in this document. The
> two quality scores are presented as **Model Alpha** and **Model Beta**.
> What follows describes the *behaviour* of the system, not the
> implementation behind it.

## What an evaluation produces

For every aligned sentence pair:

| Field | Meaning |
| --- | --- |
| `semantic_quality` | how well the target sentence preserves the meaning of the source (0–1, higher is better) |
| `reference_free_quality` | an independent quality estimate for the same pair (0–25, lower is better) |
| `source_sentiment` / `target_sentiment` | the sentiment class detected on each side |
| `sentiment_match` | whether the sentiment survived the translation |

Plus a clip-level summary (averages, sentence-pair count, sentiment
agreement) and the list of segments that could not be aligned — useful for
spotting omissions in a translation.

## How an evaluation runs

1. Both clips are transcribed into sentences.
2. Source and target sentences are aligned **by meaning**, not by position,
   so a translation that merges or splits sentences still lines up.
3. Every aligned pair is scored by the two quality scorers.
4. Sentiment is compared on both sides of every pair.
5. Anything that could not be aligned is returned separately instead of
   being dropped.

## API

Interactive documentation (Swagger UI) is served at `/docs` on the API
port, and every endpoint can be tried out from the browser.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/evaluate` | Evaluate a source/target audio pair |
| `POST` | `/transcribe` | Transcribe a single clip |
| `GET` | `/evaluations` | List stored evaluations (metadata only) |
| `GET` | `/evaluations/latest` | The most recent evaluation |
| `GET` | `/evaluations/{id}` | One full evaluation |
| `GET` | `/health` | Liveness check |

### Example request

```bash
curl -X POST "$EVAL_API_URL/evaluate" \
  -F source_audio=@source.wav \
  -F target_audio=@target.wav
```

### Example response (abridged)

```json
{
  "id": "3f9c1a2b7d04",
  "created_at": 1762000000.0,
  "transcripts": { "source": "...", "target": "..." },
  "summary": {
    "chunks": 12,
    "unmatched": 1,
    "semantic_quality": 0.812,
    "reference_free_quality": 7.42,
    "sentiment_agreement_pct": 83.3
  },
  "chunks": [
    {
      "source": "...",
      "target": "...",
      "semantic_quality": 0.88,
      "reference_free_quality": 6.1,
      "source_sentiment": "positive",
      "target_sentiment": "positive",
      "sentiment_match": true
    }
  ],
  "unmatched": { "source": [], "target": ["..."] }
}
```

Two optional query parameters on `/evaluate` tune the alignment:
`min_similarity` (how similar a pair must be to count as aligned) and
`max_merge` (how many consecutive segments may be merged on each side).

## Dashboard

The dashboard is a read-only view over the stored evaluations:

- clip-level summary tiles,
- a per-sentence-pair score chart — click a point to open that pair,
- a table of every sentence pair, with the low-scoring pairs highlighted,
- the two transcripts side by side,
- the unmatched segments, in their own expander.

A chunk that scores below **0.4** on Model Alpha is flagged, so a reviewer
can jump straight to the sentences that need attention.

## Running it

The demo is set up for a single GPU machine (a RunPod pod is the usual
choice). One script installs the dependencies, downloads the models and
starts both services:

```bash
bash API/runpod_start.sh
```

It prints the two URLs when everything is up:

```
Dashboard: https://<pod-id>-8501.proxy.runpod.net
API docs:  https://<pod-id>-8000.proxy.runpod.net/docs
```

`API/RUNPOD_SETUP.md` has the full operational guide.

## Repository layout

| Path | Role |
| --- | --- |
| `API/api.py` | The evaluation API (FastAPI) — endpoints, request/response shapes, interactive docs |
| `API/eval_core.py` | The evaluation engine: transcription, alignment, scoring, sentiment |
| `API/eval_store.py` | Persistence of evaluation results (SQLite) |
| `API/DemoApp.py` | The dashboard (Streamlit) |
| `API/runpod_start.sh` | One-command setup and start for a GPU box |
| `API/RUNPOD_SETUP.md` | Operational guide for running the demo |
| `ARCHITECTURE.md` | How the pieces fit together |
| `DemoApp.py` | Standalone copy of the dashboard for single-app deployments |

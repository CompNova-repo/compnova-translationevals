"""
FastAPI service exposing CompNova's S2ST translation evaluation engine
programmatically. See eval_core.py for the underlying pipeline (the
same one used by DemoApp.py's Streamlit UI).

Run directly:      uvicorn api:app --host 0.0.0.0 --port 8000
Interactive docs:   http://<host>:8000/docs

Client-facing note: nothing here names the models behind the scores. The
quality scores are exposed as two neutral fields ("semantic_quality" and
"reference_free_quality"), the OpenAPI text describes them generically,
and every error message is scrubbed of internal model identifiers before
it leaves the process.
"""
import os
import re
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

import eval_core

# Internal model identifiers must never reach a client, so every error
# string is passed through scrub_detail() first. Longest patterns first,
# so a specific id is replaced before its shorter fragment.
_SCRUB_RULES = [
    (r"unbabel/wmt22-cometkiwi-da", "the quality model"),
    (r"wmt22-cometkiwi-da", "the quality model"),
    (r"google/metricx-24-hybrid-large-v2p6-bfloat16", "the quality model"),
    (r"google/mt5-xl", "the quality model tokenizer"),
    (r"metricx24\.predict", "the quality scorer"),
    (r"metricx24", "the quality scorer"),
    (r"unbabel-comet", "the quality model package"),
    (r"metricx", "the quality model"),
    (r"comet[-_]?kiwi", "Model Alpha"),
    (r"comet", "Model Alpha"),
    (r"sentence[-_]transformers", "the model runtime"),
    (r"sentence-transformers/labse", "the alignment model"),
    (r"labse", "the alignment model"),
    (r"mobiuslabsgmbh/faster-whisper-large-v3-turbo", "the speech recognition model"),
    (r"faster[-_]whisper", "the speech recognition model"),
    (r"large[-_]v3[-_]turbo", "the speech recognition model"),
    (r"whisper", "the speech recognition model"),
    (r"microsoft/infoxlm-large", "an auxiliary model"),
    (r"tabularisai/multilingual-sentiment-analysis", "the sentiment model"),
    (r"tabularisai/multilingual-emotion-classification", "the emotion model"),
    (r"facebook/hf-seamless-m4t-medium", "the translation model"),
    (r"seamlessm4t", "the translation model"),
    (r"facebook/nllb-200-distilled-600m", "the translation model"),
    (r"nllb[-_]200", "the translation model"),
    (r"nllb", "the translation model"),
    (r"gtts", "the speech synthesiser"),
]


def scrub_detail(text) -> str:
    """Strip internal model identifiers out of anything a client can read."""
    out = str(text)
    for pattern, replacement in _SCRUB_RULES:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


app = FastAPI(
    title="CompNova Translation Evaluation API",
    description=(
        "Programmatic access to the S2ST translation evaluation engine: "
        "sentence-level semantic alignment (not position-based), "
        "two reference-free quality scores per chunk, and 5-class "
        "sentiment comparison (Very Negative..Very Positive)."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _warm_up_models():
    # Load every model once at startup rather than on the first request,
    # so the first real call to /evaluate isn't the one that pays for
    # every cold model load.
    eval_core.warm_up()


@app.get("/health")
def health():
    """Liveness check for container orchestration / load balancers."""
    return {"status": "ok"}


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
    return tmp.name


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio to transcribe (WAV/MP3)"),
):
    """
    Transcribe a single audio file with the speech-recognition model.
    Used by thin front ends (e.g. the dashboard's cascaded-translation path)
    that need source text to translate, WITHOUT loading the recogniser
    themselves - keeps every model loaded in exactly one place.
    """
    path = None
    try:
        path = _save_upload(audio)
        segments = eval_core.transcribe_segments(path)
        return {"transcript": eval_core.join_segment_text(segments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=scrub_detail(e))
    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.post("/evaluate")
async def evaluate(
    source_audio: UploadFile = File(..., description="Source-language audio (WAV/MP3)"),
    target_audio: UploadFile = File(..., description="Translated audio to evaluate (WAV/MP3)"),
    min_similarity: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Minimum semantic similarity for a chunk to count as aligned",
    ),
    max_merge: int = Query(
        2, ge=1, le=3,
        description="Max consecutive segments to try merging on each side (2 = allows 1-1, 1-2, 2-1, 2-2)",
    ),
):
    """
    Evaluate a source/translated audio pair end to end:

    1. Transcribes both files (sentence-level segments)
    2. Aligns chunks by MEANING using a multilingual sentence-embedding
       model + merge-aware dynamic programming, not by list position
    3. Scores every aligned chunk with two reference-free quality models
       ("semantic_quality", 0-1 higher is better; "reference_free_quality",
       0-25 lower is better)
    4. Runs 5-class sentiment on both sides of every chunk
    5. Returns per-chunk detail plus clip-level summary averages,
       and any segments that couldn't be aligned (potential omissions)
    """
    src_path = mt_path = None
    try:
        src_path = _save_upload(source_audio)
        mt_path = _save_upload(target_audio)
        result = eval_core.evaluate_pair(
            src_path, mt_path, min_similarity=min_similarity, max_merge=max_merge
        )
        eval_id, created_at = save_evaluation(
            result, source_audio.filename, target_audio.filename
        )
        return JSONResponse(content={"id": eval_id, "created_at": created_at, **result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=scrub_detail(e))
    finally:
        for p in (src_path, mt_path):
            if p and os.path.exists(p):
                os.remove(p)


from eval_store import init_db, save_evaluation, list_evaluations, get_evaluation, get_latest
init_db()

@app.get("/evaluations")
def evaluations(limit: int = 20):
    return list_evaluations(limit)

@app.get("/evaluations/latest")
def latest_evaluation():
    record = get_latest()
    if record is None:
        raise HTTPException(404, "No evaluations yet")
    return record

@app.get("/evaluations/{eval_id}")
def evaluation(eval_id: str):
    record = get_evaluation(eval_id)
    if record is None:
        raise HTTPException(404, "Evaluation not found")
    return record

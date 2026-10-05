"""
FastAPI service exposing CompNova's S2ST translation evaluation engine
programmatically. See eval_core.py for the underlying pipeline (the
same one used by DemoApp.py's Streamlit UI).

Run directly:      uvicorn api:app --host 0.0.0.0 --port 8000
Interactive docs:   http://<host>:8000/docs
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

import eval_core

app = FastAPI(
    title="CompNova Translation Evaluation API",
    description=(
        "Programmatic access to the S2ST translation evaluation engine: "
        "sentence-level semantic alignment (not position-based), "
        "COMET-Kiwi + MetricX-24 quality scoring per chunk, and 5-class "
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
    Transcribe a single audio file (Faster-Whisper). Used by thin
    front ends (e.g. the Streamlit demo's cascaded-translation path)
    that need source text to translate, WITHOUT loading Whisper
    themselves - keeps every model loaded in exactly one place.
    """
    path = None
    try:
        path = _save_upload(audio)
        segments = eval_core.transcribe_segments(path)
        return {"transcript": eval_core.join_segment_text(segments)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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

    1. Transcribes both files (Faster-Whisper, sentence-level segments)
    2. Aligns chunks by MEANING using LaBSE embeddings + merge-aware
       dynamic programming, not by list position
    3. Scores every aligned chunk with COMET-Kiwi and MetricX-24
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
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (src_path, mt_path):
            if p and os.path.exists(p):
                os.remove(p)

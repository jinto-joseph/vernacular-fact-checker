"""
api.py
------
FastAPI inference server for the Vernacular Fact-Checker pipeline.

Endpoints:
    GET  /health          - liveness check
    POST /predict         - single-post fact-check
    POST /predict-batch   - batch fact-check (parallel)
    POST /predict-image   - OCR text extraction + fact-check from image
    POST /analyze-image   - full image analysis: tamper detection + deepfake detection

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Interactive docs:
    http://127.0.0.1:8000/docs
"""

import os
import tempfile
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from main import extract_text_from_image, process_batch, process_post
from image_analysis import analyze_image

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Automated Fact-Checker API",
    version="1.1.0",
    description=(
        "High-throughput fake-news fact-checking pipeline. "
        "Strips non-factual noise, extracts claims, retrieves verified facts, "
        "and returns verdicts with confidence scores. "
        "Also supports image tamper detection and deepfake classification."
    ),
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw social media or news post text to fact-check.",
        examples=["Breaking news: RBI is shutting all banks nationwide tomorrow"],
    )


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of raw texts to process in parallel.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    summary="Liveness check",
    response_description="Returns status ok when the service is running.",
)
def health() -> Dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post(
    "/predict",
    summary="Fact-check a single post",
    response_description="Full pipeline output including verdict, confidence, and ML prediction.",
)
def predict(request: PredictRequest) -> Dict[str, Any]:
    """
    Run the full fact-checking pipeline on a single text post.

    Returns preprocessing metrics, the extracted claim, retrieval result,
    verdict, confidence score, and optional ML classification.
    """
    return process_post(request.text)


@app.post(
    "/predict-batch",
    summary="Fact-check multiple posts in parallel",
    response_description="Count and list of per-post pipeline results.",
)
def predict_batch(request: BatchPredictRequest) -> Dict[str, Any]:
    """
    Run the pipeline concurrently on a list of posts.

    Worker count is automatically scaled to the batch size (max 8).
    """
    workers = min(8, max(1, len(request.texts)))
    results = process_batch(request.texts, workers=workers)
    return {"count": len(results), "results": results}


@app.post(
    "/predict-image",
    summary="Fact-check a post from an image via OCR",
    response_description="Same pipeline output as /predict, sourced from OCR-extracted text.",
)
def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Extract text from an uploaded image via OCR, then run the fact-checking pipeline.

    Best for: screenshots of news articles, WhatsApp forwards, infographics with text.
    Requires pytesseract and pillow.

    Supported formats: JPEG, PNG, BMP, TIFF.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        extracted_text = extract_text_from_image(tmp_path)
        return process_post(extracted_text, is_image=False)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post(
    "/analyze-image",
    summary="Full image analysis: tamper detection + deepfake classification",
    response_description="image_flagged flag plus detailed tamper and deepfake analysis.",
)
def analyze_image_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Run tamper detection (ELA + EXIF) and deepfake classification on an uploaded image.

    Best for: edited photos, manipulated images, AI-generated faces, deepfakes.

    Returns:
        image_flagged     - True if any check triggered
        tamper_analysis   - ELA score, tamper_detected flag, has_camera_metadata flag
        deepfake_analysis - deepfake_score, deepfake_detected flag, model_available flag
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        return analyze_image(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

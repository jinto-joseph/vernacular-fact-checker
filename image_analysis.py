"""
image_analysis.py
-----------------
Image-based misinformation detection module.

Covers four image types:
    1. Screenshots / WhatsApp forwards  -> OCR via pytesseract (main.py handles this)
    2. Infographics with false stats    -> OCR via pytesseract (main.py handles this)
    3. Edited / manipulated images      -> ELA + EXIF tamper detection (this module)
    4. Deepfakes / AI-generated faces   -> HuggingFace classifier (this module)

Usage:
    from image_analysis import analyze_image
    result = analyze_image("path/to/image.jpg")
"""

import io
import importlib.util
from functools import lru_cache
from typing import Dict, Any

from PIL import Image, ImageChops, ImageEnhance

# ---------------------------------------------------------------------------
# Dependency checks — checked at runtime, not import time
# ---------------------------------------------------------------------------
def _check_transformers() -> bool:
    """Safely verify both transformers and torch are importable."""
    try:
        if not importlib.util.find_spec("transformers"):
            return False
        if not importlib.util.find_spec("torch"):
            return False
        import torch        # noqa: F401
        import transformers # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tamper detection — Error Level Analysis (ELA) + EXIF
# ---------------------------------------------------------------------------
def detect_manipulation(image_path: str) -> Dict[str, Any]:
    """
    Detect image manipulation using Error Level Analysis (ELA).

    ELA works by re-saving the image at a known JPEG quality and comparing
    it against the original. Regions that were edited or composited show
    higher error levels because they were re-compressed at a different quality.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys:
            tamper_detected (bool)       - True if ELA score exceeds threshold
            ela_score (float)            - Average brightness of ELA diff (0-255 scale)
            has_camera_metadata (bool)   - True if EXIF has Make/Model/DateTime
    """
    original = Image.open(image_path).convert("RGB")

    # ELA: re-save at quality=90 and compare pixel differences
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(original, recompressed)
    enhanced = ImageEnhance.Brightness(diff).enhance(20)
    pixels = list(enhanced.getdata())
    avg_ela = sum(sum(p) for p in pixels) / (len(pixels) * 3)

    # EXIF check — legitimate camera photos usually have Make/Model/DateTime
    has_camera_metadata = False
    try:
        from PIL.ExifTags import TAGS
        exif_data = original._getexif() or {}
        camera_tags = {"Make", "Model", "DateTime"}
        has_camera_metadata = any(TAGS.get(k) in camera_tags for k in exif_data)
    except Exception:
        pass

    return {
        "tamper_detected": avg_ela > 15,
        "ela_score": round(avg_ela, 2),
        "has_camera_metadata": has_camera_metadata,
    }


# ---------------------------------------------------------------------------
# Deepfake detection — HuggingFace pretrained classifier
# ---------------------------------------------------------------------------
_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
_DEEPFAKE_THRESHOLD = 0.7


@lru_cache(maxsize=1)
def _load_deepfake_detector():
    """
    Load the deepfake detection pipeline (cached after first call).
    Returns None if transformers or torch are unavailable or fail to load.
    """
    if not _check_transformers():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "image-classification",
            model=_DEEPFAKE_MODEL_ID,
        )
    except Exception:
        return None


def detect_deepfake(image_path: str) -> Dict[str, Any]:
    """
    Classify whether a face image is a deepfake or AI-generated.

    Uses a pretrained HuggingFace image-classification model.
    Falls back gracefully if transformers/torch are not installed or fail.

    Args:
        image_path: Path to image file containing a face.

    Returns:
        dict with keys:
            deepfake_detected (bool)   - True if top label is fake and score > threshold
            deepfake_score (float)     - Confidence score (0.0-1.0)
            model_available (bool)     - False if dependencies missing or load failed
    """
    try:
        detector = _load_deepfake_detector()
    except Exception:
        detector = None

    if detector is None:
        return {
            "deepfake_detected": False,
            "deepfake_score": 0.0,
            "model_available": False,
        }

    try:
        results = detector(image_path)
        top = results[0]
        return {
            "deepfake_detected": (
                top["label"].lower() == "fake" and top["score"] > _DEEPFAKE_THRESHOLD
            ),
            "deepfake_score": round(top["score"], 3),
            "model_available": True,
        }
    except Exception as exc:
        return {
            "deepfake_detected": False,
            "deepfake_score": 0.0,
            "model_available": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Unified image analysis
# ---------------------------------------------------------------------------
def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Run all image analysis checks on a single image file.

    Combines:
        - Tamper detection (ELA + EXIF)
        - Deepfake detection (HuggingFace classifier)

    Note: OCR text extraction is handled separately in main.py via
    extract_text_from_image(), which feeds into the fact-checking pipeline.

    Args:
        image_path: Path to the image file.

    Returns:
        Merged dict with all results and a top-level image_flagged bool.
    """
    tamper = detect_manipulation(image_path)
    deepfake = detect_deepfake(image_path)

    return {
        "image_flagged": tamper["tamper_detected"] or deepfake["deepfake_detected"],
        "tamper_analysis": tamper,
        "deepfake_analysis": deepfake,
    }
"""
image_analysis.py
-----------------
Image-based misinformation detection module.

Covers four image types:
    1. Screenshots / WhatsApp forwards  -> OCR via pytesseract (main.py handles this)
    2. Infographics with false stats    -> OCR via pytesseract (main.py handles this)
    3. Edited / manipulated images      -> ELA + EXIF tamper detection (this module)
    4. Deepfakes / AI-generated faces   -> HuggingFace classifier (this module)

ELA threshold calibration:
    0  - 25  : likely unedited (normal JPEG compression artifacts)
    25 - 40  : suspicious — possible light editing
    40+      : strong manipulation signal

The threshold is set at 35 (not 15) to avoid false positives on real photos.
EXIF metadata absence alone is NOT enough to flag — screenshots and
phone-exported images legitimately lack camera EXIF.
"""

import io
import importlib.util
from functools import lru_cache
from typing import Dict, Any

from PIL import Image, ImageChops, ImageEnhance

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
TRANSFORMERS_AVAILABLE = False


def _check_transformers() -> bool:
    global TRANSFORMERS_AVAILABLE
    try:
        if not importlib.util.find_spec("transformers"):
            return False
        if not importlib.util.find_spec("torch"):
            return False
        import torch        # noqa: F401
        import transformers # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception:
        return False


# ELA thresholds — calibrated to reduce false positives on real photos
_ELA_THRESHOLD_CLEAN     = 25   # below this = almost certainly unedited
_ELA_THRESHOLD_SUSPICIOUS = 35  # above this = likely manipulated
_ELA_THRESHOLD_TAMPERED   = 50  # above this = strong manipulation signal


def _ela_verdict(score: float, has_exif: bool) -> tuple:
    """
    Combine ELA score and EXIF presence into a tamper verdict.

    Returns (tamper_detected: bool, confidence: str)
    """
    if score < _ELA_THRESHOLD_CLEAN:
        return False, "clean"
    if score < _ELA_THRESHOLD_SUSPICIOUS:
        # Borderline — only flag if EXIF is also missing
        return (not has_exif), "borderline"
    if score < _ELA_THRESHOLD_TAMPERED:
        return True, "suspicious"
    return True, "high"


# ---------------------------------------------------------------------------
# Tamper detection — Error Level Analysis (ELA) + EXIF
# ---------------------------------------------------------------------------
def detect_manipulation(image_path: str) -> Dict[str, Any]:
    """
    Detect image manipulation using Error Level Analysis (ELA) + EXIF.

    ELA works by re-saving the image at a known JPEG quality and comparing
    against the original. Edited regions show higher error because they were
    compressed at a different quality level.

    Threshold is set conservatively at 35 (not 15) to avoid false positives
    on real unedited photos which naturally score 10–25 due to JPEG artifacts.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys:
            tamper_detected (bool)       — True if ELA + EXIF signals confirm manipulation
            ela_score (float)            — Raw ELA score (0–255 scale)
            ela_level (str)              — clean / borderline / suspicious / high
            has_camera_metadata (bool)   — True if EXIF has Make/Model/DateTime
            confidence (str)             — Explanation of the verdict
    """
    original = Image.open(image_path).convert("RGB")

    # ELA: re-save at quality=92 and diff against original
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=92)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(original, recompressed)
    enhanced = ImageEnhance.Brightness(diff).enhance(20)
    pixels = list(enhanced.getdata())
    avg_ela = sum(sum(p) for p in pixels) / (len(pixels) * 3)

    # EXIF check
    has_camera_metadata = False
    try:
        from PIL.ExifTags import TAGS
        exif_data = original._getexif() or {}
        camera_tags = {"Make", "Model", "DateTime"}
        has_camera_metadata = any(TAGS.get(k) in camera_tags for k in exif_data)
    except Exception:
        pass

    tamper_detected, ela_level = _ela_verdict(avg_ela, has_camera_metadata)

    confidence_messages = {
        "clean": "ELA score is within normal JPEG compression range. Image appears unedited.",
        "borderline": "ELA score is slightly elevated. No camera metadata found. Treat with mild caution.",
        "suspicious": "ELA score indicates possible editing. Regions may have been composited or altered.",
        "high": "High ELA score strongly indicates manipulation. Image has likely been significantly edited.",
    }

    return {
        "tamper_detected": tamper_detected,
        "ela_score": round(avg_ela, 2),
        "ela_level": ela_level,
        "has_camera_metadata": has_camera_metadata,
        "confidence": confidence_messages[ela_level],
    }


# ---------------------------------------------------------------------------
# Deepfake detection — HuggingFace pretrained classifier
# ---------------------------------------------------------------------------
_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
_DEEPFAKE_THRESHOLD = 0.7


@lru_cache(maxsize=1)
def _load_deepfake_detector():
    if not _check_transformers():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline("image-classification", model=_DEEPFAKE_MODEL_ID)
    except Exception:
        return None


def detect_deepfake(image_path: str) -> Dict[str, Any]:
    """
    Classify whether a face image is a deepfake or AI-generated.

    Falls back gracefully if transformers/torch are not installed.

    Args:
        image_path: Path to image file containing a face.

    Returns:
        dict with keys:
            deepfake_detected (bool)
            deepfake_score (float)
            model_available (bool)
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
            "note": "Deepfake model unavailable in this environment. Deploy with GPU for full support.",
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
            "note": str(exc),
        }


# ---------------------------------------------------------------------------
# Unified image analysis
# ---------------------------------------------------------------------------
def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Run all image analysis checks on a single image file.

    Combines ELA tamper detection and deepfake classification.
    image_flagged is True only when tamper_detected OR deepfake_detected.

    Note: OCR text extraction for screenshots/infographics is handled
    separately in main.py via extract_text_from_image().

    Args:
        image_path: Path to the image file.

    Returns:
        Merged dict with all results and a top-level image_flagged bool.
    """
    tamper = detect_manipulation(image_path)
    deepfake = detect_deepfake(image_path)

    image_flagged = tamper["tamper_detected"] or deepfake["deepfake_detected"]

    return {
        "image_flagged": image_flagged,
        "summary": (
            "Image appears authentic." if not image_flagged
            else "Image has been flagged — possible manipulation or AI generation detected."
        ),
        "tamper_analysis": tamper,
        "deepfake_analysis": deepfake,
    }

"""
main.py
-------
End-to-end Vernacular Fact-Checker pipeline.

Stages:
    1. Ingestion        — raw text or OCR from image
    2. Optimization     — ScaleDown AI compression + rule-based cleaning
    3. Claim Extraction — lightweight heuristic
    4. Fact Retrieval   — Tavily Search → NewsData.io → local store
    5. Verification     — verdict + confidence score
    6. ML Classification — optional trained classifier

Run demo:
    python main.py
"""

import importlib
import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Sequence

from preprocessing import clean_text, reduction_stats
from fact_retrieval import retrieve_fact

OCR_AVAILABLE: bool = (
    bool(importlib.util.find_spec("pytesseract"))
    and bool(importlib.util.find_spec("PIL"))
)


# ---------------------------------------------------------------------------
# ML model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_ml_model(model_path: str):
    """
    Load a trained scikit-learn pipeline from disk.
    Checks multiple candidate paths in priority order.
    Returns None if no artifact is found.
    """
    candidate_paths = [
        model_path,
        "artifacts/best_fake_news_model.joblib",
        "artifacts/fake_news_model.joblib",
    ]
    resolved = next((p for p in candidate_paths if os.path.exists(p)), None)
    if resolved is None:
        return None
    joblib = importlib.import_module("joblib")
    return joblib.load(resolved)


def classify_fake_news_ml(
    text: str,
    model_path: str = "artifacts/fake_news_model.joblib",
) -> Dict[str, object]:
    """
    Run ML fake-news classification on raw text.
    Returns available=False if no trained artifact is found.
    """
    model = _load_ml_model(model_path)
    if model is None:
        return {"available": False, "prediction": "N/A", "model_path": model_path}
    prediction = model.predict([clean_text(text)])[0]
    return {"available": True, "prediction": str(prediction), "model_path": model_path}


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract OCR.

    Raises:
        RuntimeError: If pytesseract or pillow are not installed.
        FileNotFoundError: If the image file does not exist.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR dependencies not installed. Run: pip install pytesseract pillow")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    pytesseract = importlib.import_module("pytesseract")
    pil_image = importlib.import_module("PIL.Image")
    return pytesseract.image_to_string(pil_image.open(image_path))


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------
def claim_extraction(cleaned_text: str) -> str:
    """
    Extract the most likely factual claim using a longest-sentence heuristic.
    Keeps this stage lightweight for high-throughput processing.
    """
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    if not sentences:
        return cleaned_text
    return max(sentences, key=lambda s: len(s.split()))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_claim(claim: str) -> Dict[str, object]:
    """
    Retrieve best matching fact and produce a verdict + confidence score.

    Uses fact_retrieval.retrieve_fact() which queries in priority order:
        1. Tavily Search API  — real-time web search across fact-check domains
        2. NewsData.io API    — real-time Indian news cross-reference
        3. Local fact store   — 12 hardcoded Indian misinformation patterns
        4. Unverified fallback
    """
    retrieved = retrieve_fact(claim)
    score = float(retrieved["retrieval_score"])
    verdict = str(retrieved["base_verdict"])

    if verdict == "Unverified":
        confidence = 0.35
    elif score < 0.2:
        verdict = "Misleading"
        confidence = round(min(0.99, max(0.4, 0.45 + score)), 3)
    else:
        confidence = round(min(0.99, max(0.4, 0.45 + score)), 3)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "matched_fact_id": retrieved["fact_id"],
        "matched_fact": retrieved["fact_text"],
        "retrieval_score": score,
        "source": retrieved.get("source", ""),
        "source_url": retrieved.get("source_url", ""),
        "rating": retrieved.get("rating", ""),
    }


# ---------------------------------------------------------------------------
# Per-post pipeline
# ---------------------------------------------------------------------------
def process_post(input_data: str, is_image: bool = False) -> Dict[str, object]:
    """
    Run the full fact-checking pipeline on a single post.

    Args:
        input_data: Raw text string, or image path when is_image=True.
        is_image:   If True, runs OCR on input_data before processing.

    Returns:
        Full pipeline output dictionary including all stage results.
    """
    raw_text = extract_text_from_image(input_data) if is_image else input_data
    cleaned = clean_text(raw_text)
    claim = claim_extraction(cleaned)
    verification = verify_claim(claim)
    ml_result = classify_fake_news_ml(raw_text)

    return {
        "input": input_data,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "claim": claim,
        "preprocessing_metrics": reduction_stats(raw_text, cleaned),
        "verification": verification,
        "ml_classification": ml_result,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_batch(posts: Sequence[str], workers: int = 8) -> List[Dict[str, object]]:
    """
    Process a list of posts concurrently using a thread pool.

    Args:
        posts:   Sequence of raw text strings.
        workers: Number of parallel worker threads (default: 8).

    Returns:
        List of per-post result dicts, each annotated with pipeline_latency_ms.
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_post, posts))
    elapsed = time.perf_counter() - start
    per_post_ms = round((elapsed / max(1, len(posts))) * 1000.0, 2)
    for item in results:
        item["pipeline_latency_ms"] = per_post_ms
    return results


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def benchmark_pipeline(sample_posts: Sequence[str], runs: int = 3) -> Dict[str, float]:
    """
    Measure throughput, latency, token reduction, and estimated cost savings.

    Args:
        sample_posts: Posts to benchmark against.
        runs:         Number of repeated runs for stable averages.

    Returns:
        Dictionary of aggregate benchmark metrics.
    """
    total_chars_before = total_chars_after = 0
    total_tokens_before = total_tokens_after = 0
    total_elapsed = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        results = process_batch(sample_posts)
        total_elapsed += time.perf_counter() - start
        for r in results:
            m = r["preprocessing_metrics"]
            total_chars_before += int(m["original_chars"])
            total_chars_after += int(m["cleaned_chars"])
            total_tokens_before += int(m["original_tokens"])
            total_tokens_after += int(m["cleaned_tokens"])

    processed = len(sample_posts) * runs
    avg_latency_ms = (total_elapsed / max(1, processed)) * 1000.0
    throughput_per_min = (processed / max(1e-9, total_elapsed)) * 60.0
    token_reduction_pct = (total_tokens_before - total_tokens_after) / max(1, total_tokens_before) * 100.0
    char_reduction_pct = (total_chars_before - total_chars_after) / max(1, total_chars_before) * 100.0
    cost_per_token = 0.000002
    base_cost = total_tokens_before * cost_per_token
    optimized_cost = total_tokens_after * cost_per_token

    return {
        "items_processed": float(processed),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_posts_per_min": round(throughput_per_min, 2),
        "token_reduction_pct": round(token_reduction_pct, 2),
        "char_reduction_pct": round(char_reduction_pct, 2),
        "estimated_cost_before_usd": round(base_cost, 4),
        "estimated_cost_after_usd": round(optimized_cost, 4),
        "estimated_cost_savings_pct": round(
            (base_cost - optimized_cost) / max(1e-9, base_cost) * 100.0, 2
        ),
        "scaledown_active": bool(os.getenv("SCALEDOWN_API_KEY")),
        "tavily_active": bool(os.getenv("TAVILY_API_KEY")),
        "newsdata_active": bool(os.getenv("NEWSDATA_API_KEY")),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_POSTS = [
    "BREAKING!!! Share this now!!! Every citizen gets 5000 rupees daily from tonight 😱😱",
    "Election dates out now, check official Election Commission website for final schedule.",
    "Forward this viral update: RBI shutting all banks nationwide tomorrow!!!",
    "Heatwave alert in north India this week. Follow IMD advisories.",
    "OMG watch till end this shocking policy update wow wow wow",
]


def demo() -> None:
    """Run the pipeline on curated demo posts and print a benchmark summary."""
    results = process_batch(DEMO_POSTS)

    print("=" * 60)
    print("  VERNACULAR FACT-CHECKER  |  Demo Run")
    print("=" * 60)

    for i, item in enumerate(results, start=1):
        v = item["verification"]
        ml = item["ml_classification"]
        print(f"\n[{i}] Claim   : {item['claim']}")
        print(f"    Verdict : {v['verdict']} (confidence={v['confidence']}, score={v['retrieval_score']})")
        print(f"    Source  : {v.get('source', 'local')}")
        if ml["available"]:
            print(f"    ML      : {ml['prediction']}")

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    summary = benchmark_pipeline(DEMO_POSTS, runs=3)
    for key, value in summary.items():
        print(f"  {key:<40} {value}")


if __name__ == "__main__":
    demo()
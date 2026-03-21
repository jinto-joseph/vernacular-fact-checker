"""
main.py
-------
End-to-end fact-checking pipeline.

Stages:
    1. Ingestion  – raw text or OCR from image
    2. Optimization – strip non-factual noise (preprocessing.py)
    3. Claim Extraction – lightweight heuristic, no heavy model dependency
    4. Fact Retrieval – tag-overlap with LRU cache
    5. Verification – verdict + confidence score
    6. ML Classification – optional trained classifier (artifacts/)

Run demo:
    python main.py
"""

import importlib
import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

from preprocessing import clean_text, reduction_stats

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------
OCR_AVAILABLE: bool = (
    bool(importlib.util.find_spec("pytesseract"))
    and bool(importlib.util.find_spec("PIL"))
)


# ---------------------------------------------------------------------------
# Verified fact store (demo; swap with FAISS/vector DB for production)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifiedFact:
    fact_id: str
    text: str
    verdict: str
    tags: tuple  # immutable so instances are hashable


VERIFIED_FACTS: List[VerifiedFact] = [
    VerifiedFact(fact_id="F001", text="No nationwide bank closure was announced by RBI in 2026.", verdict="False", tags=("rbi", "bank", "banks", "closure", "close", "nationwide", "2026", "reserve")),
    VerifiedFact(fact_id="F002", text="India has no policy that gives every citizen 5000 rupees per day.", verdict="False", tags=("india", "policy", "citizen", "citizens", "5000", "rupees", "per", "day", "money", "cash", "free")),
    VerifiedFact(fact_id="F003", text="Heatwaves can happen in March in multiple Indian states.", verdict="True", tags=("heatwave", "heat", "march", "indian", "india", "states", "weather", "temperature", "imd", "alert")),
    VerifiedFact(fact_id="F004", text="The Election Commission publishes official polling schedules on its portal.", verdict="True", tags=("election", "elections", "commission", "official", "schedule", "polling", "portal", "vote", "voting", "dates")),
    VerifiedFact(fact_id="F005", text="Government schemes are announced through official notifications, not random forwards.", verdict="Misleading", tags=("government", "scheme", "schemes", "official", "notification", "forward", "whatsapp", "viral", "message")),
    VerifiedFact(fact_id="F006", text="Viral death claims about public figures are frequently false and unverified.", verdict="Misleading", tags=("dead", "death", "died", "killed", "passed", "away", "alive", "hoax", "fake", "no", "more")),
    VerifiedFact(fact_id="F007", text="No new demonetisation or currency ban has been officially announced in India.", verdict="False", tags=("demonetisation", "demonetization", "currency", "note", "notes", "ban", "banned", "invalid", "rupee", "rs")),
    VerifiedFact(fact_id="F008", text="Internet shutdowns in India are officially notified by state governments.", verdict="Misleading", tags=("internet", "shutdown", "network", "mobile", "data", "blocked", "ban", "suspended", "offline")),
    VerifiedFact(fact_id="F009", text="Unverified claims about free government giveaways are commonly circulated misinformation.", verdict="Misleading", tags=("free", "giveaway", "gift", "scheme", "apply", "link", "click", "register", "form", "benefit", "subsidy")),
    VerifiedFact(fact_id="F010", text="Fuel prices in India are revised periodically by oil marketing companies.", verdict="Misleading", tags=("petrol", "diesel", "fuel", "price", "prices", "reduced", "free", "cheap", "litre", "oil")),
    VerifiedFact(fact_id="F011", text="Health advisories should be verified through official government or WHO sources.", verdict="Misleading", tags=("covid", "virus", "vaccine", "disease", "health", "medicine", "cure", "hospital", "doctor", "outbreak")),
    VerifiedFact(fact_id="F012", text="Military operations and border situations are officially communicated by the Ministry of Defence.", verdict="Misleading", tags=("army", "military", "war", "attack", "border", "soldier", "soldiers", "china", "pakistan", "strike", "operation")),
]


# ---------------------------------------------------------------------------
# ML model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_ml_model(model_path: str):
    """
    Load a trained scikit-learn pipeline from disk.

    Checks multiple candidate paths in priority order.
    Returns None if no artifact is found (pipeline degrades gracefully).
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

    Args:
        text:       Raw input text (preprocessing applied internally).
        model_path: Primary path to check for a trained .joblib artifact.

    Returns:
        dict with keys: available (bool), prediction (str), model_path (str).
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
    Extract text from an image file using pytesseract.

    Args:
        image_path: Absolute or relative path to the image file.

    Raises:
        RuntimeError:    If OCR dependencies are not installed.
        FileNotFoundError: If the image file does not exist.

    Returns:
        Extracted text string.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR dependencies not installed. "
            "Run: pip install pytesseract pillow"
        )
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
    Extract the most likely factual claim from cleaned text.

    Strategy: prefer the longest sentence (by word count), which tends to
    carry the most factual content. Keeps this stage lightweight for
    high-throughput processing.

    Args:
        cleaned_text: Preprocessed/optimized text.

    Returns:
        Best candidate claim string.
    """
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    if not sentences:
        return cleaned_text
    return max(sentences, key=lambda s: len(s.split()))


# ---------------------------------------------------------------------------
# Fact retrieval (tag-overlap, LRU-cached)
# ---------------------------------------------------------------------------
def _overlap_score(claim_tokens: Sequence[str], fact_tags: tuple) -> float:
    """Jaccard-style overlap between claim tokens and fact tags."""
    claim_set = set(claim_tokens)
    fact_set = set(fact_tags)
    if not claim_set or not fact_set:
        return 0.0
    return len(claim_set & fact_set) / len(fact_set)


@lru_cache(maxsize=2048)
def retrieve_fact_for_claim(claim: str) -> Dict[str, object]:
    """
    Retrieve the best-matching verified fact for a given claim.

    Uses token-overlap scoring with LRU caching to handle repeated/viral
    claims efficiently without redundant computation.

    Args:
        claim: Extracted factual claim string.

    Returns:
        dict with keys: fact_id, fact_text, base_verdict, retrieval_score.
    """
    claim_tokens = claim.split()
    best_score, best_fact = max(
        ((_overlap_score(claim_tokens, fact.tags), fact) for fact in VERIFIED_FACTS),
        key=lambda x: x[0],
    )
    return {
        "fact_id": best_fact.fact_id,
        "fact_text": best_fact.text,
        "base_verdict": best_fact.verdict,
        "retrieval_score": round(best_score, 3),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_claim_against_fact(
    claim: str,
    retrieved: Dict[str, object],
) -> Dict[str, object]:
    """
    Produce a final verdict and confidence score.

    Low retrieval scores (< 0.2) indicate the fact store has no strong match,
    so the verdict falls back to "Misleading" to avoid false confidence.

    Args:
        claim:     Extracted claim (unused directly, kept for future NLI).
        retrieved: Output of retrieve_fact_for_claim().

    Returns:
        dict with keys: verdict, confidence, matched_fact_id,
                        matched_fact, retrieval_score.
    """
    score = float(retrieved["retrieval_score"])
    verdict = str(retrieved["base_verdict"])

    # No match at all — do not show a misleading fact, return unverified
    if score == 0.0:
        return {
            "verdict": "Unverified",
            "confidence": 0.35,
            "matched_fact_id": "N/A",
            "matched_fact": "No matching fact found in the verified fact store for this claim.",
            "retrieval_score": score,
        }

    # Weak match — downgrade to Misleading
    if score < 0.2:
        verdict = "Misleading"

    confidence = round(min(0.99, max(0.4, 0.45 + score)), 3)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "matched_fact_id": retrieved["fact_id"],
        "matched_fact": retrieved["fact_text"],
        "retrieval_score": score,
    }


# ---------------------------------------------------------------------------
# Per-post pipeline
# ---------------------------------------------------------------------------
def process_post(input_data: str, is_image: bool = False) -> Dict[str, object]:
    """
    Run the full fact-checking pipeline on a single post.

    Args:
        input_data: Raw text string, or path to an image file when is_image=True.
        is_image:   If True, runs OCR on input_data before processing.

    Returns:
        Dictionary containing every pipeline stage's output.
    """
    raw_text = extract_text_from_image(input_data) if is_image else input_data
    cleaned = clean_text(raw_text)
    claim = claim_extraction(cleaned)
    retrieved = retrieve_fact_for_claim(claim)
    verification = verify_claim_against_fact(claim, retrieved)
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
def process_batch(
    posts: Sequence[str],
    workers: int = 8,
) -> List[Dict[str, object]]:
    """
    Process a list of posts concurrently using a thread pool.

    Args:
        posts:   Sequence of raw text strings.
        workers: Number of parallel worker threads.

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
def benchmark_pipeline(
    sample_posts: Sequence[str],
    runs: int = 3,
) -> Dict[str, float]:
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

        for result in results:
            m = result["preprocessing_metrics"]
            total_chars_before += int(m["original_chars"])
            total_chars_after += int(m["cleaned_chars"])
            total_tokens_before += int(m["original_tokens"])
            total_tokens_after += int(m["cleaned_tokens"])

    processed = len(sample_posts) * runs
    avg_latency_ms = (total_elapsed / max(1, processed)) * 1000.0
    throughput_per_min = (processed / max(1e-9, total_elapsed)) * 60.0

    token_reduction_pct = (
        (total_tokens_before - total_tokens_after) / max(1, total_tokens_before) * 100.0
    )
    char_reduction_pct = (
        (total_chars_before - total_chars_after) / max(1, total_chars_before) * 100.0
    )

    # Cost simulation assumes $0.002 per 1000 tokens (illustrative)
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
    }


# ---------------------------------------------------------------------------
# Demo entry point
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
        print(f"\n[{i}] Claim : {item['claim']}")
        print(
            f"    Verdict     : {v['verdict']} "
            f"(confidence={v['confidence']}, retrieval={v['retrieval_score']})"
        )
        if ml["available"]:
            print(f"    ML Prediction: {ml['prediction']}")
        else:
            print("    ML Prediction: model not trained — run compare_models.py first")

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY  (5 runs)")
    print("=" * 60)
    summary = benchmark_pipeline(DEMO_POSTS, runs=5)
    for key, value in summary.items():
        print(f"  {key:<35} {value}")


if __name__ == "__main__":
    demo()

"""
fact_retrieval.py
-----------------
Multi-source fact retrieval with priority ordering:

    1. Google Fact Check Tools API  — verified claims from AltNews, AFP, Snopes etc.
    2. MediaStack News API          — real-time news cross-reference
    3. Local fact store             — 12 hardcoded Indian misinformation patterns
    4. Unverified fallback          — honest no-match response

All sources use LRU caching to handle repeated viral claims efficiently
without redundant API calls, satisfying the throughput requirement.

Environment variables required:
    GOOGLE_FACTCHECK_API_KEY  — Google Cloud API key with Fact Check Tools enabled
    MEDIASTACK_API_KEY        — MediaStack API key (free tier: 500 req/month)
"""

import os
from functools import lru_cache
from typing import Dict, Any, List
import urllib.request
import urllib.parse
import json

from preprocessing import clean_text


# ---------------------------------------------------------------------------
# Local verified fact store (fallback)
# ---------------------------------------------------------------------------
VERIFIED_FACTS = [
    {"fact_id": "F001", "text": "No nationwide bank closure was announced by RBI in 2026.", "verdict": "False", "tags": {"rbi", "bank", "banks", "closure", "close", "nationwide", "reserve"}},
    {"fact_id": "F002", "text": "India has no policy that gives every citizen 5000 rupees per day.", "verdict": "False", "tags": {"india", "policy", "citizen", "citizens", "5000", "rupees", "money", "cash", "free"}},
    {"fact_id": "F003", "text": "Heatwaves can happen in March in multiple Indian states.", "verdict": "True", "tags": {"heatwave", "heat", "march", "indian", "india", "states", "weather", "temperature", "imd"}},
    {"fact_id": "F004", "text": "The Election Commission publishes official polling schedules on its portal.", "verdict": "True", "tags": {"election", "elections", "commission", "official", "schedule", "polling", "vote", "voting"}},
    {"fact_id": "F005", "text": "Government schemes are announced through official notifications, not random forwards.", "verdict": "Misleading", "tags": {"government", "scheme", "schemes", "official", "notification", "forward", "whatsapp", "viral"}},
    {"fact_id": "F006", "text": "Viral death claims about public figures are frequently false and unverified.", "verdict": "Misleading", "tags": {"dead", "death", "died", "killed", "passed", "away", "alive", "hoax", "fake"}},
    {"fact_id": "F007", "text": "No new demonetisation or currency ban has been officially announced in India.", "verdict": "False", "tags": {"demonetisation", "demonetization", "currency", "note", "notes", "ban", "banned", "invalid", "rupee"}},
    {"fact_id": "F008", "text": "Internet shutdowns in India are officially notified by state governments.", "verdict": "Misleading", "tags": {"internet", "shutdown", "network", "mobile", "data", "blocked", "ban", "suspended"}},
    {"fact_id": "F009", "text": "Unverified claims about free government giveaways are commonly circulated misinformation.", "verdict": "Misleading", "tags": {"free", "giveaway", "gift", "scheme", "apply", "link", "click", "register", "benefit", "subsidy"}},
    {"fact_id": "F010", "text": "Fuel prices in India are revised periodically by oil marketing companies.", "verdict": "Misleading", "tags": {"petrol", "diesel", "fuel", "price", "prices", "reduced", "cheap", "litre", "oil"}},
    {"fact_id": "F011", "text": "Health advisories should be verified through official government or WHO sources.", "verdict": "Misleading", "tags": {"covid", "virus", "vaccine", "disease", "health", "medicine", "cure", "hospital", "doctor", "outbreak"}},
    {"fact_id": "F012", "text": "Military operations and border situations are officially communicated by the Ministry of Defence.", "verdict": "Misleading", "tags": {"army", "military", "war", "attack", "border", "soldier", "soldiers", "china", "pakistan", "strike"}},
]


def _map_rating_to_verdict(rating: str) -> str:
    """Map a free-text fact-check rating to our verdict system."""
    r = rating.lower()
    if any(w in r for w in ("false", "incorrect", "wrong", "fabricated", "fake", "no truth", "misleads", "pants on fire", "baseless")):
        return "False"
    if any(w in r for w in ("true", "correct", "accurate", "verified", "real", "confirmed")):
        return "True"
    return "Misleading"


# ---------------------------------------------------------------------------
# Source 1: Google Fact Check Tools API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def _google_fact_check(claim: str) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
    if not api_key:
        return {}
    try:
        params = urllib.parse.urlencode({"query": claim, "key": api_key, "languageCode": "en"})
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        claims = data.get("claims", [])
        if not claims:
            return {}

        top = claims[0]
        review = top.get("claimReview", [{}])[0]
        rating = review.get("textualRating", "")

        return {
            "fact_id": "GFCT",
            "fact_text": review.get("title", top.get("text", claim)),
            "base_verdict": _map_rating_to_verdict(rating),
            "retrieval_score": 0.95,
            "source": review.get("publisher", {}).get("name", "Google Fact Check"),
            "source_url": review.get("url", ""),
            "rating": rating,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Source 2: MediaStack real-time news API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _mediastack_news_check(claim: str) -> Dict[str, Any]:
    api_key = os.getenv("MEDIASTACK_API_KEY", "")
    if not api_key:
        return {}
    try:
        keywords = " ".join(claim.split()[:6])
        params = urllib.parse.urlencode({
            "access_key": api_key,
            "keywords": keywords,
            "countries": "in",
            "languages": "en",
            "limit": 3,
            "sort": "published_desc",
        })
        url = f"http://api.mediastack.com/v1/news?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        articles = data.get("data", [])
        if not articles:
            return {}

        top = articles[0]
        return {
            "fact_id": "NEWS",
            "fact_text": top.get("title", ""),
            "base_verdict": "True",
            "retrieval_score": 0.7,
            "source": top.get("source", "MediaStack News"),
            "source_url": top.get("url", ""),
            "rating": "News source corroboration",
            "published_at": top.get("published_at", ""),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Source 3: Local fact store (tag-overlap)
# ---------------------------------------------------------------------------
def _local_fact_check(claim: str) -> Dict[str, Any]:
    claim_tokens = set(claim.lower().split())
    best_score = 0.0
    best_fact = None

    for fact in VERIFIED_FACTS:
        tags = fact["tags"]
        if not claim_tokens or not tags:
            continue
        overlap = len(claim_tokens & tags) / len(tags)
        if overlap > best_score:
            best_score = overlap
            best_fact = fact

    if best_score == 0.0 or best_fact is None:
        return {}

    return {
        "fact_id": best_fact["fact_id"],
        "fact_text": best_fact["text"],
        "base_verdict": best_fact["verdict"],
        "retrieval_score": round(best_score, 3),
        "source": "Local fact store",
        "source_url": "",
        "rating": best_fact["verdict"],
    }


# ---------------------------------------------------------------------------
# Unified retrieval (priority order)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def retrieve_fact(claim: str) -> Dict[str, Any]:
    """
    Retrieve the best-matching verified fact using all available sources.

    Priority order:
        1. Google Fact Check Tools API (real verified facts)
        2. MediaStack News API (real-time news corroboration)
        3. Local fact store (rule-based Indian misinformation patterns)
        4. Unverified fallback

    Args:
        claim: Extracted and preprocessed factual claim.

    Returns:
        Best available fact match with verdict, confidence, and source info.
    """
    # 1. Google Fact Check
    result = _google_fact_check(claim)
    if result:
        return result

    # 2. MediaStack news cross-reference
    result = _mediastack_news_check(claim)
    if result:
        return result

    # 3. Local fact store
    result = _local_fact_check(claim)
    if result:
        return result

    # 4. Unverified fallback
    return {
        "fact_id": "N/A",
        "fact_text": "No matching verified fact found for this claim.",
        "base_verdict": "Unverified",
        "retrieval_score": 0.0,
        "source": "None",
        "source_url": "",
        "rating": "",
    }

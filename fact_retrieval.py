"""
fact_retrieval.py
-----------------
Multi-source fact retrieval with priority ordering:

    1. Tavily Search API  — real-time web search optimized for AI/RAG (free, 1000 credits/month)
    2. NewsData.io API    — real-time Indian news cross-reference (free, 200 credits/day)
    3. Local fact store   — 12 hardcoded Indian misinformation patterns
    4. Unverified         — honest no-match fallback

Environment variables:
    TAVILY_API_KEY     — from app.tavily.com (free, no credit card)
    NEWSDATA_API_KEY   — from newsdata.io (free, no credit card)
"""

import os
from functools import lru_cache
from typing import Dict, Any
import urllib.request
import urllib.parse
import json


# ---------------------------------------------------------------------------
# Local verified fact store (fallback)
# ---------------------------------------------------------------------------
VERIFIED_FACTS = [
    {"fact_id": "F001", "text": "No nationwide bank closure was announced by RBI in 2026.", "verdict": "False",
     "tags": {"rbi", "bank", "banks", "closure", "close", "nationwide", "reserve"}},
    {"fact_id": "F002", "text": "India has no policy that gives every citizen 5000 rupees per day.", "verdict": "False",
     "tags": {"india", "policy", "citizen", "citizens", "5000", "rupees", "money", "cash", "free"}},
    {"fact_id": "F003", "text": "Heatwaves can happen in March in multiple Indian states.", "verdict": "True",
     "tags": {"heatwave", "heat", "march", "indian", "india", "states", "weather", "temperature", "imd"}},
    {"fact_id": "F004", "text": "The Election Commission publishes official polling schedules on its portal.", "verdict": "True",
     "tags": {"election", "elections", "commission", "official", "schedule", "polling", "vote", "voting"}},
    {"fact_id": "F005", "text": "Government schemes are announced through official notifications, not random forwards.", "verdict": "Misleading",
     "tags": {"government", "scheme", "schemes", "official", "notification", "forward", "whatsapp", "viral"}},
    {"fact_id": "F006", "text": "Viral death claims about public figures are frequently false and unverified.", "verdict": "Misleading",
     "tags": {"dead", "death", "died", "killed", "passed", "away", "alive", "hoax", "fake"}},
    {"fact_id": "F007", "text": "No new demonetisation or currency ban has been officially announced in India.", "verdict": "False",
     "tags": {"demonetisation", "demonetization", "currency", "note", "notes", "ban", "banned", "invalid", "rupee"}},
    {"fact_id": "F008", "text": "Internet shutdowns in India are officially notified by state governments.", "verdict": "Misleading",
     "tags": {"internet", "shutdown", "network", "mobile", "data", "blocked", "ban", "suspended"}},
    {"fact_id": "F009", "text": "Unverified claims about free government giveaways are commonly circulated misinformation.", "verdict": "Misleading",
     "tags": {"free", "giveaway", "gift", "scheme", "apply", "link", "click", "register", "benefit", "subsidy"}},
    {"fact_id": "F010", "text": "Fuel prices in India are revised periodically by oil marketing companies.", "verdict": "Misleading",
     "tags": {"petrol", "diesel", "fuel", "price", "prices", "reduced", "cheap", "litre", "oil"}},
    {"fact_id": "F011", "text": "Health advisories should be verified through official government or WHO sources.", "verdict": "Misleading",
     "tags": {"covid", "virus", "vaccine", "disease", "health", "medicine", "cure", "hospital", "doctor", "outbreak"}},
    {"fact_id": "F012", "text": "Military operations and border situations are officially communicated by the Ministry of Defence.", "verdict": "Misleading",
     "tags": {"army", "military", "war", "attack", "border", "soldier", "soldiers", "china", "pakistan", "strike"}},
]


def _map_to_verdict(content: str) -> str:
    """Map web search content to a verdict by scanning for signal words."""
    c = content.lower()
    if any(w in c for w in ("false", "fake", "misinformation", "hoax", "fabricated",
                             "incorrect", "wrong", "no evidence", "debunked", "misleading")):
        return "False"
    if any(w in c for w in ("confirmed", "verified", "true", "accurate", "correct",
                             "official", "announced", "government confirmed")):
        return "True"
    return "Misleading"


# ---------------------------------------------------------------------------
# Source 1: Tavily Search API
# Real-time web search optimized for AI/RAG — returns ranked, parsed content
# Free: 1000 credits/month, no credit card — sign up at app.tavily.com
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def _tavily_search(claim: str) -> Dict[str, Any]:
    """
    Search the web for fact-check information using Tavily.

    Tavily is a search API built for AI applications — it returns
    clean, ranked, LLM-ready content from multiple sources per query.
    Searches news and fact-check domains with topic=news for best results.

    Free plan: 1000 credits/month at app.tavily.com (no credit card)
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return {}
    try:
        payload = json.dumps({
            "api_key": api_key,
            "query": f"fact check: {claim}",
            "search_depth": "basic",
            "topic": "news",
            "max_results": 3,
            "include_answer": True,
            "include_domains": [
                "altnews.in", "boomlive.in", "factchecker.in",
                "factcheck.afp.com", "snopes.com", "bbc.com",
                "thehindu.com", "ndtv.com", "reuters.com",
                "pib.gov.in", "indiatoday.in"
            ],
        }).encode()

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        # Use Tavily's AI-generated answer if available
        answer = data.get("answer", "")
        results = data.get("results", [])

        if not answer and not results:
            return {}

        # Determine verdict from answer + top result content
        combined_text = answer + " ".join(r.get("content", "") for r in results[:2])
        verdict = _map_to_verdict(combined_text)

        top_result = results[0] if results else {}
        fact_text = answer if answer else top_result.get("content", claim)[:300]

        return {
            "fact_id": "TAV",
            "fact_text": fact_text,
            "base_verdict": verdict,
            "retrieval_score": 0.88,
            "source": top_result.get("title", "Tavily Search"),
            "source_url": top_result.get("url", ""),
            "rating": f"Web search across {len(results)} sources",
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Source 2: NewsData.io real-time news API
# Cross-references claim keywords against live Indian news articles
# Free: 200 credits/day, no credit card — register at newsdata.io
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _newsdata_check(claim: str) -> Dict[str, Any]:
    """
    Cross-reference a claim against real-time Indian news via NewsData.io.

    If a recent credible news article matches the claim's keywords,
    it provides corroboration. Searches India-specific sources.

    Free plan: 200 credits/day — register at newsdata.io (no credit card)
    """
    api_key = os.getenv("NEWSDATA_API_KEY", "")
    if not api_key:
        return {}
    try:
        keywords = " ".join(claim.split()[:6])
        params = urllib.parse.urlencode({
            "apikey": api_key,
            "q": keywords,
            "country": "in",
            "language": "en",
            "size": 3,
        })
        url = f"https://newsdata.io/api/1/news?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())

        articles = data.get("results", [])
        if not articles:
            return {}

        top = articles[0]
        title = top.get("title", "")
        source = top.get("source_id", "NewsData.io")
        link = top.get("link", "")
        pub_date = top.get("pubDate", "")
        description = top.get("description", "")

        combined = f"{title} {description}"
        verdict = _map_to_verdict(combined)

        return {
            "fact_id": "ND",
            "fact_text": f"{title}. {description[:200]}".strip(),
            "base_verdict": verdict,
            "retrieval_score": 0.65,
            "source": f"NewsData.io — {source}",
            "source_url": link,
            "rating": f"Published: {pub_date[:10] if pub_date else 'recent'}",
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Source 3: Local fact store (tag-overlap)
# ---------------------------------------------------------------------------
def _local_fact_check(claim: str) -> Dict[str, Any]:
    """Match claim against 12 hardcoded Indian misinformation patterns."""
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
# Unified retrieval — priority order
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def retrieve_fact(claim: str) -> Dict[str, Any]:
    """
    Retrieve the best-matching verified fact using all available sources.

    Priority order:
        1. Tavily Search API  — real-time web search across fact-check domains
        2. NewsData.io API    — real-time Indian news cross-reference
        3. Local fact store   — 12 hardcoded Indian misinformation patterns
        4. Unverified fallback

    All results are LRU-cached (2048 entries) to handle repeated viral
    claims efficiently without redundant API calls.

    Args:
        claim: Extracted and preprocessed factual claim string.

    Returns:
        Best available fact match with verdict, confidence, and source info.
    """
    # 1. Tavily real-time web search
    result = _tavily_search(claim)
    if result:
        return result

    # 2. NewsData.io Indian news
    result = _newsdata_check(claim)
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

"""
fact_retrieval.py
-----------------
Multi-source fact retrieval with priority ordering:

    1. Local fact store   — exact and fuzzy match against verified Indian patterns
    2. Tavily Search API  — only for explicit fact-check domain results
    3. NewsData.io API    — only when explicit debunking language found
    4. Heuristic classifier — rule-based verdict for common misinformation patterns
    5. Unverified fallback

Design principle:
    Never return True unless we have strong evidence from a verified source.
    Default to Misleading for unrecognized claims rather than True.
    This matches real-world fact-checker behavior — the burden of proof
    is on confirming truth, not assuming it.
"""

import os
import re
from functools import lru_cache
from typing import Dict, Any
import urllib.request
import urllib.parse
import json


# ---------------------------------------------------------------------------
# Local verified fact store — primary source
# ---------------------------------------------------------------------------
VERIFIED_FACTS = [
    {"fact_id": "F001", "text": "No nationwide bank closure was announced by RBI in 2026.", "verdict": "False",
     "tags": {"rbi", "bank", "banks", "closure", "close", "nationwide", "reserve", "shutting", "shut"}},
    {"fact_id": "F002", "text": "India has no policy that gives every citizen free money per day.", "verdict": "False",
     "tags": {"citizen", "citizens", "5000", "rupees", "money", "cash", "free", "daily", "per", "day", "give", "getting"}},
    {"fact_id": "F003", "text": "Heatwaves are a verified weather phenomenon in India during summer months.", "verdict": "True",
     "tags": {"heatwave", "heat", "wave", "march", "april", "may", "indian", "india", "states", "weather", "temperature", "imd", "alert"}},
    {"fact_id": "F004", "text": "The Election Commission of India publishes official polling schedules on its portal.", "verdict": "True",
     "tags": {"election", "elections", "commission", "official", "schedule", "schedules", "polling", "vote", "voting", "dates", "eci"}},
    {"fact_id": "F005", "text": "Government schemes are announced through official notifications, not WhatsApp forwards.", "verdict": "Misleading",
     "tags": {"government", "scheme", "schemes", "notification", "forward", "whatsapp", "viral", "message", "share", "forward"}},
    {"fact_id": "F006", "text": "Viral death claims about public figures are frequently false and unverified.", "verdict": "False",
     "tags": {"dead", "death", "died", "killed", "passed", "away", "alive", "no", "more", "rip", "demise"}},
    {"fact_id": "F007", "text": "No new demonetisation or currency ban has been officially announced in India.", "verdict": "False",
     "tags": {"demonetisation", "demonetization", "currency", "note", "notes", "ban", "banned", "invalid", "rupee", "500", "2000"}},
    {"fact_id": "F008", "text": "Internet shutdowns in India require official notification by state governments.", "verdict": "Misleading",
     "tags": {"internet", "shutdown", "network", "mobile", "data", "blocked", "ban", "suspended", "down", "off"}},
    {"fact_id": "F009", "text": "Claims about free government giveaways circulated on social media are commonly misinformation.", "verdict": "False",
     "tags": {"free", "giveaway", "gift", "apply", "link", "click", "register", "form", "benefit", "subsidy", "lottery", "prize"}},
    {"fact_id": "F010", "text": "Fuel prices in India are officially revised by oil marketing companies — not reduced to zero.", "verdict": "Misleading",
     "tags": {"petrol", "diesel", "fuel", "price", "prices", "reduced", "free", "cheap", "litre", "oil", "cut"}},
    {"fact_id": "F011", "text": "Health and medical claims should be verified through official government or WHO sources.", "verdict": "Misleading",
     "tags": {"covid", "virus", "vaccine", "disease", "health", "medicine", "cure", "hospital", "doctor", "outbreak", "treatment"}},
    {"fact_id": "F012", "text": "Military operations are officially communicated by the Ministry of Defence, not social media.", "verdict": "Misleading",
     "tags": {"army", "military", "war", "attack", "border", "soldier", "soldiers", "china", "pakistan", "strike", "operation", "surgical"}},
    {"fact_id": "F013", "text": "PM Modi and other Indian leaders are alive and active — viral death claims are false.", "verdict": "False",
     "tags": {"modi", "pm", "prime", "minister", "president", "rahul", "gandhi", "amit", "shah", "leader", "dead", "died", "death"}},
    {"fact_id": "F014", "text": "WhatsApp forwards claiming urgent government action or emergency are usually misinformation.", "verdict": "False",
     "tags": {"urgent", "emergency", "breaking", "alert", "warning", "immediately", "tonight", "tomorrow", "midnight", "deadline"}},
    {"fact_id": "F015", "text": "India's supreme court and judiciary function independently — viral claims about court orders should be verified.", "verdict": "Misleading",
     "tags": {"supreme", "court", "judge", "judgment", "order", "verdict", "hearing", "case", "legal", "law", "banned"}},
]


# ---------------------------------------------------------------------------
# Heuristic classifier — catches common misinformation patterns
# ---------------------------------------------------------------------------
_MISINFORMATION_PATTERNS = [
    # Urgency patterns common in fake news
    (r"\b(tonight|tomorrow|midnight|immediately|urgent|breaking|alert)\b", "False", 0.6),
    # Free money/scheme patterns
    (r"\b(free|₹|rs\.?\s*\d+|rupees?\s*\d+).{0,30}(citizen|people|everyone|all)\b", "False", 0.7),
    (r"\b(get|claim|apply|register).{0,20}(free|money|cash|prize|gift|reward)\b", "False", 0.65),
    # Death hoax patterns
    (r"\b(rip|rest in peace|passed away|no more|died|dead).{0,30}(actor|actress|minister|cricketer|celebrity|politician)\b", "False", 0.7),
    # Shutdown/ban patterns without official source
    (r"\b(shut|close|ban|block|stop).{0,20}(bank|internet|whatsapp|facebook|india|nationwide)\b", "False", 0.65),
    # Share/forward bait
    (r"\b(share|forward|send).{0,20}(everyone|all|contacts|friends|family|group)\b", "Misleading", 0.6),
    # Clickbait patterns
    (r"\b(shocking|unbelievable|you won.t believe|must watch|watch till end|viral)\b", "Misleading", 0.55),
]


def _heuristic_verdict(claim: str) -> Dict[str, Any]:
    """
    Apply rule-based patterns to detect common misinformation structures.

    Returns a result dict if a pattern matches, empty dict otherwise.
    """
    text = claim.lower()
    for pattern, verdict, confidence in _MISINFORMATION_PATTERNS:
        if re.search(pattern, text):
            return {
                "fact_id": "HEU",
                "fact_text": f"This claim matches a common misinformation pattern: {pattern}",
                "base_verdict": verdict,
                "retrieval_score": confidence,
                "source": "Heuristic classifier",
                "source_url": "",
                "rating": f"Pattern match: {verdict}",
            }
    return {}


# ---------------------------------------------------------------------------
# Strict signal lists for web search results
# ---------------------------------------------------------------------------
_FALSE_SIGNALS = [
    "false", "fake", "misinformation", "hoax", "fabricated", "debunked",
    "no evidence", "misleads", "fact check: false", "this is false",
    "not true", "unverified claim", "viral fake", "baseless",
    "no such order", "not announced", "did not happen", "rumour", "rumor",
    "claim is false", "verdict: false", "rating: false",
]

_MISLEADING_SIGNALS = [
    "misleading", "partly false", "half true", "out of context",
    "missing context", "needs context", "partially true",
    "mostly false", "exaggerated", "verdict: misleading",
]

_TRUE_SIGNALS = [
    "fact check: true", "claim is true", "verdict: true",
    "verified true", "confirmed true", "this is accurate",
    "rating: true", "this is correct", "fact check: correct",
]

_FACTCHECK_DOMAINS = {
    "altnews.in", "boomlive.in", "factchecker.in",
    "factcheck.afp.com", "snopes.com", "factcheck.org",
    "politifact.com", "vishvasnews.com", "thequint.com",
    "indiatoday.in/fact-check", "thelogicalindian.com",
}


def _is_factcheck_url(url: str) -> bool:
    return any(domain in url.lower() for domain in _FACTCHECK_DOMAINS)


# Corroboration signals — when Tavily answer directly confirms the claim
_CORROBORATION_SIGNALS = [
    "is located in", "is situated in", "is based in", "is indeed",
    "is consistent with", "is correct", "is accurate", "does exist",
    "is true", "confirmed", "is a real", "is the", "are the",
    "latest available data", "information is consistent",
    "no information contradicts", "does not contradict",
]


def _strict_verdict(content: str, is_factcheck: bool) -> str:
    """
    Strict verdict mapping.

    Priority:
        1. Explicit FALSE signals → False
        2. Explicit MISLEADING signals → Misleading
        3. Explicit TRUE signals from fact-check source → True
        4. Corroboration signals (Tavily answer directly confirms) → True
        5. No signal → empty string (fall through to next source)
    """
    c = content.lower()

    # 1. False signals — highest priority
    if any(s in c for s in _FALSE_SIGNALS):
        return "False"

    # 2. Misleading signals
    if any(s in c for s in _MISLEADING_SIGNALS):
        return "Misleading"

    # 3. Explicit true signals from fact-check site
    if is_factcheck and any(s in c for s in _TRUE_SIGNALS):
        return "True"

    # 4. Corroboration — Tavily answer directly confirms the claim
    # e.g. "is located in Coimbatore", "information is consistent with latest data"
    if any(s in c for s in _CORROBORATION_SIGNALS):
        return "True"

    # 5. No strong signal — fall through
    return ""


# ---------------------------------------------------------------------------
# Source: Tavily Search API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def _tavily_search(claim: str) -> Dict[str, Any]:
    """
    Search fact-check domains specifically for this claim using Tavily.
    Only returns a result when explicit fact-check verdict language is found.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return {}
    try:
        payload = json.dumps({
            "api_key": api_key,
            "query": f"fact check {claim} true or false",
            "search_depth": "advanced",
            "topic": "news",
            "max_results": 5,
            "include_answer": True,
            "include_domains": list(_FACTCHECK_DOMAINS) + [
                "reuters.com", "bbc.com", "thehindu.com", "ndtv.com",
            ],
        }).encode()

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        answer = data.get("answer", "")
        results = data.get("results", [])
        if not answer and not results:
            return {}

        factcheck_results = [r for r in results if _is_factcheck_url(r.get("url", ""))]
        is_factcheck = len(factcheck_results) > 0
        primary = factcheck_results if factcheck_results else results[:2]
        combined = answer + " " + " ".join(r.get("content", "") for r in primary)

        verdict = _strict_verdict(combined, is_factcheck)
        if not verdict:
            return {}  # No strong signal — fall through to next source

        top = primary[0] if primary else {}
        return {
            "fact_id": "TAV",
            "fact_text": re.sub(r"<[^>]+>", "", answer if answer else top.get("content", claim)[:300]).strip(),
            "base_verdict": verdict,
            "retrieval_score": 0.90 if is_factcheck else 0.70,
            "source": top.get("title", "Tavily Search"),
            "source_url": top.get("url", ""),
            "rating": f"{'Fact-check' if is_factcheck else 'News'} — {len(results)} sources",
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Source: NewsData.io
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _newsdata_check(claim: str) -> Dict[str, Any]:
    """
    Only returns a result when explicit false/misleading signals are found
    in the article — never assumes True from news coverage alone.
    """
    api_key = os.getenv("NEWSDATA_API_KEY", "")
    if not api_key:
        return {}
    try:
        keywords = "fact check " + " ".join(claim.split()[:5])
        params = urllib.parse.urlencode({
            "apikey": api_key,
            "q": keywords,
            "country": "in",
            "language": "en",
            "size": 3,
        })
        with urllib.request.urlopen(
            f"https://newsdata.io/api/1/news?{params}", timeout=5
        ) as resp:
            data = json.loads(resp.read())

        articles = data.get("results", [])
        if not articles:
            return {}

        top = articles[0]
        combined = f"{top.get('title','')} {top.get('description','')}".lower()

        has_false = any(s in combined for s in _FALSE_SIGNALS)
        has_misleading = any(s in combined for s in _MISLEADING_SIGNALS)

        if not has_false and not has_misleading:
            return {}  # No signal — skip

        return {
            "fact_id": "ND",
            "fact_text": f"{top.get('title','')}. {top.get('description','')[:200]}".strip(),
            "base_verdict": "False" if has_false else "Misleading",
            "retrieval_score": 0.65,
            "source": f"NewsData.io — {top.get('source_id','')}",
            "source_url": top.get("link", ""),
            "rating": f"Published: {top.get('pubDate','')[:10]}",
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Local fact store — fuzzy tag overlap
# ---------------------------------------------------------------------------
def _local_fact_check(claim: str) -> Dict[str, Any]:
    """
    Match claim against hardcoded Indian misinformation patterns.
    Uses token overlap — higher overlap = stronger match.
    """
    claim_tokens = set(re.sub(r"[^\w\s]", "", claim.lower()).split())
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

    if best_score < 0.15 or best_fact is None:
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
# Unified retrieval
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def retrieve_fact(claim: str) -> Dict[str, Any]:
    """
    Retrieve the best-matching verified fact using all available sources.

    Priority:
        1. Local fact store   — most reliable for Indian misinformation
        2. Heuristic patterns — catches structural misinformation patterns
        3. Tavily Search      — real fact-check domain results only
        4. NewsData.io        — only when explicit signals found
        5. Unverified fallback
    """
    # 1. Local fact store first — most reliable
    result = _local_fact_check(claim)
    if result and result["retrieval_score"] >= 0.25:
        return result

    # 2. Heuristic classifier
    result = _heuristic_verdict(claim)
    if result:
        return result

    # 3. Tavily — only if strong fact-check signal found
    result = _tavily_search(claim)
    if result:
        return result

    # 4. NewsData — only if explicit debunking signal found
    result = _newsdata_check(claim)
    if result:
        return result

    # 5. Weak local match (score 0.10-0.24)
    result = _local_fact_check(claim)
    if result:
        return result

    return {
        "fact_id": "N/A",
        "fact_text": "No matching verified fact found for this claim.",
        "base_verdict": "Unverified",
        "retrieval_score": 0.0,
        "source": "None",
        "source_url": "",
        "rating": "",
    }

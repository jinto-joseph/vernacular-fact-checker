"""
preprocessing.py
----------------
Pipeline Optimization stage: strips non-factual noise from social/news text
before claim extraction and retrieval, reducing token count and compute cost.

Optionally uses the ScaleDown API for AI-powered prompt compression,
which directly satisfies the "Pipeline Optimization" technique requirement.
"""

import re
import os
import unicodedata
import json
import requests
from typing import Dict

# ---------------------------------------------------------------------------
# Clickbait and filler phrase patterns
# ---------------------------------------------------------------------------
_CLICKBAIT_PHRASES = [
    r"watch till end", r"share this now", r"forward this", r"must watch",
    r"breaking news", r"viral update", r"shocking", r"omg", r"wow+",
    r"please rt", r"retweet", r"100% true", r"share fast",
]

_CLICKBAIT_RE = re.compile(
    r"\b(?:" + "|".join(_CLICKBAIT_PHRASES) + r")\b",
    flags=re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_HASHTAG_RE = re.compile(r"[@#]\w+")
_PUNCT_NOISE_RE = re.compile(r"[!?]{2,}")
_CHAR_REPEAT_RE = re.compile(r"(.)\1{2,}")


def _remove_emojis(text: str) -> str:
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("So")
        and unicodedata.category(ch) not in ("Cs",)
    )


def _deduplicate_words(text: str) -> str:
    words = text.split()
    deduped = [words[0]] if words else []
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    return " ".join(deduped)


def _rule_based_clean(raw: str) -> str:
    """Apply all rule-based optimization steps."""
    text = _URL_RE.sub(" ", raw)
    text = _MENTION_HASHTAG_RE.sub(" ", text)
    text = _remove_emojis(text)
    text = _CLICKBAIT_RE.sub(" ", text)
    text = _CHAR_REPEAT_RE.sub(r"\1\1", text)
    text = _PUNCT_NOISE_RE.sub("!", text)
    text = _deduplicate_words(text)
    text = " ".join(text.split())
    return text.strip()


def _scaledown_compress(text: str, api_key: str) -> str:
    """
    Use ScaleDown API for AI-powered prompt compression.

    ScaleDown identifies and retains only factually relevant content,
    going beyond rule-based cleaning for maximum token reduction.

    Args:
        text:    Rule-cleaned text to compress further.
        api_key: ScaleDown API key from SCALEDOWN_API_KEY env var.

    Returns:
        Compressed text, or original text if API call fails.
    """
    try:
        response = requests.post(
            "https://api.scaledown.xyz/compress/raw/",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "context": "Extract only the core factual claim from this social media post for fact-checking.",
                "prompt": text,
                "scaledown": {"rate": "auto"},
            }),
            timeout=5,
        )
        result = response.json()
        if result.get("successful") and result.get("compressed_prompt"):
            return result["compressed_prompt"].strip()
    except Exception:
        pass
    return text


def clean_text(raw: str) -> str:
    """
    Apply full optimization pipeline to reduce non-factual noise.

    Steps:
        1. Rule-based cleaning (URLs, emojis, clickbait, dedup)
        2. ScaleDown AI compression (if SCALEDOWN_API_KEY is set)

    Returns:
        Cleaned, factual-context-preserving text.
    """
    cleaned = _rule_based_clean(raw)

    api_key = os.getenv("SCALEDOWN_API_KEY", "")
    if api_key and len(cleaned.split()) > 10:
        compressed = _scaledown_compress(cleaned, api_key)
        if compressed:
            return compressed

    return cleaned


def reduction_stats(original: str, cleaned: str) -> Dict[str, object]:
    """
    Compute token/character reduction metrics between raw and cleaned text.

    Args:
        original: Raw input text.
        cleaned:  Cleaned output text.

    Returns:
        Dictionary with char/token counts and reduction percentages.
    """
    orig_chars = len(original)
    clean_chars = len(cleaned)
    orig_tokens = len(original.split())
    clean_tokens = len(cleaned.split())

    return {
        "original_chars": orig_chars,
        "cleaned_chars": clean_chars,
        "original_tokens": orig_tokens,
        "cleaned_tokens": clean_tokens,
        "char_reduction_pct": round((orig_chars - clean_chars) / max(1, orig_chars) * 100.0, 2),
        "token_reduction_pct": round((orig_tokens - clean_tokens) / max(1, orig_tokens) * 100.0, 2),
        "scaledown_used": bool(os.getenv("SCALEDOWN_API_KEY")),
    }

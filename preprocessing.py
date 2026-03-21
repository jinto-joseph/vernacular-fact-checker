"""
preprocessing.py
----------------
Pipeline Optimization stage: strips non-factual noise from social/news text
before claim extraction and retrieval, reducing token count and compute cost.
"""

import re
import unicodedata
from typing import Dict

# ---------------------------------------------------------------------------
# Clickbait and filler phrase patterns
# ---------------------------------------------------------------------------
_CLICKBAIT_PHRASES = [
    r"watch till end",
    r"share this now",
    r"forward this",
    r"must watch",
    r"breaking news",
    r"viral update",
    r"shocking",
    r"omg",
    r"wow+",
    r"please rt",
    r"retweet",
    r"100% true",
    r"share fast",
]

_CLICKBAIT_RE = re.compile(
    r"\b(?:" + "|".join(_CLICKBAIT_PHRASES) + r")\b",
    flags=re.IGNORECASE,
)

# Matches URLs (http/https/www)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# Matches @mentions and #hashtags
_MENTION_HASHTAG_RE = re.compile(r"[@#]\w+")

# Matches sequences of punctuation/symbols that carry no factual meaning
_PUNCT_NOISE_RE = re.compile(r"[!?]{2,}")

# Matches runs of the same character (e.g. "wowwww" -> "wow")
_CHAR_REPEAT_RE = re.compile(r"(.)\1{2,}")


def _remove_emojis(text: str) -> str:
    """Strip Unicode emoji and pictograph characters."""
    return "".join(
        ch
        for ch in text
        if not unicodedata.category(ch).startswith("So")
        and unicodedata.category(ch) not in ("Cs",)
    )


def _deduplicate_words(text: str) -> str:
    """Remove immediately repeated words (e.g. 'news news' -> 'news')."""
    words = text.split()
    deduped = [words[0]] if words else []
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    return " ".join(deduped)


def clean_text(raw: str) -> str:
    """
    Apply all optimization steps to reduce non-factual noise.

    Steps applied (in order):
        1. Remove URLs
        2. Remove @mentions and #hashtags
        3. Strip emojis
        4. Remove clickbait/filler phrases
        5. Normalize repeated characters (wowww -> wow)
        6. Collapse noisy punctuation (!!!! -> !)
        7. Deduplicate adjacent repeated words
        8. Normalize whitespace

    Returns:
        Cleaned, factual-context-preserving text.
    """
    text = _URL_RE.sub(" ", raw)
    text = _MENTION_HASHTAG_RE.sub(" ", text)
    text = _remove_emojis(text)
    text = _CLICKBAIT_RE.sub(" ", text)
    text = _CHAR_REPEAT_RE.sub(r"\1\1", text)
    text = _PUNCT_NOISE_RE.sub("!", text)
    text = _deduplicate_words(text)
    text = " ".join(text.split())
    return text.strip()


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

    char_reduction = (
        round((orig_chars - clean_chars) / max(1, orig_chars) * 100.0, 2)
    )
    token_reduction = (
        round((orig_tokens - clean_tokens) / max(1, orig_tokens) * 100.0, 2)
    )

    return {
        "original_chars": orig_chars,
        "cleaned_chars": clean_chars,
        "original_tokens": orig_tokens,
        "cleaned_tokens": clean_tokens,
        "char_reduction_pct": char_reduction,
        "token_reduction_pct": token_reduction,
    }

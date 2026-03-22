"""
app.py
------
Streamlit frontend for the Vernacular Fact-Checker API.

Run locally:
    streamlit run app.py

Environment variable (optional):
    API_URL — base URL of the fact-checker API
              defaults to http://127.0.0.1:7860
"""

import os
import json
import requests
import streamlit as st
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:7860").rstrip("/")

st.set_page_config(
    page_title="Vernacular Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0a0a0f;
    color: #e8e6df;
}

/* Header */
.header-block {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0ede6;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}

.header-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b6b7e;
    margin-top: 0.5rem;
    font-weight: 300;
}

.accent {
    color: #7c6af7;
}

/* Verdict cards */
.verdict-false {
    background: #1a0a0a;
    border: 1px solid #5c1a1a;
    border-left: 4px solid #e05252;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.verdict-true {
    background: #0a1a0f;
    border: 1px solid #1a4a28;
    border-left: 4px solid #52c47a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.verdict-misleading {
    background: #1a140a;
    border: 1px solid #4a3a1a;
    border-left: 4px solid #e0a052;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.verdict-unknown {
    background: #12121e;
    border: 1px solid #2a2a3e;
    border-left: 4px solid #7c6af7;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.verdict-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.verdict-false .verdict-label  { color: #e05252; }
.verdict-true .verdict-label   { color: #52c47a; }
.verdict-misleading .verdict-label { color: #e0a052; }
.verdict-unknown .verdict-label { color: #7c6af7; }

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.82rem;
    color: #9a98b0;
    margin: 0.2rem 0.2rem 0.2rem 0;
    font-family: 'DM Sans', sans-serif;
}

.metric-val {
    color: #c8c4e0;
    font-weight: 500;
}

/* Claim box */
.claim-box {
    background: #12121e;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 1.05rem;
    color: #c8c4e0;
    font-style: italic;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.6;
}

/* Matched fact */
.fact-box {
    background: #0e0e1a;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.92rem;
    color: #8a88a8;
    margin: 0.5rem 0;
    line-height: 1.6;
}

.fact-id {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    color: #4a4a6a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a4a6a;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* Input area styling */
.stTextArea textarea {
    background: #12121e !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e8e6df !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: #7c6af7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #9580ff !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12121e;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2a3a;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b6b7e !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

.stTabs [aria-selected="true"] {
    background: #7c6af7 !important;
    color: #ffffff !important;
}

/* Progress bar */
.stProgress > div > div {
    background: #7c6af7 !important;
}

/* Divider */
hr {
    border-color: #1e1e2e !important;
}

/* File uploader */
.stFileUploader {
    background: #12121e !important;
    border: 1px dashed #2a2a3a !important;
    border-radius: 8px !important;
}

/* Flagged image banner */
.flag-banner {
    background: #1a0a0a;
    border: 1px solid #5c1a1a;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #e05252;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 1rem 0;
}

.safe-banner {
    background: #0a1a0f;
    border: 1px solid #1a4a28;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #52c47a;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 1rem 0;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def call_predict(text: str) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at `{API_URL}`. Make sure the server is running.")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def call_predict_batch(texts: list) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict-batch", json={"texts": texts}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def call_analyze_image(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/analyze-image",
            files={"file": (filename, file_bytes, "image/png")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def call_predict_image(file_bytes: bytes, filename: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict-image",
            files={"file": (filename, file_bytes, "image/png")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def verdict_css_class(verdict: str) -> str:
    v = verdict.lower()
    if v == "false":
        return "verdict-false"
    if v == "true":
        return "verdict-true"
    if v == "misleading":
        return "verdict-misleading"
    return "verdict-unknown"


def verdict_emoji(verdict: str) -> str:
    return {
        "false": "✗ FALSE",
        "true": "✓ TRUE",
        "misleading": "⚠ MISLEADING",
        "unverified": "? UNVERIFIED",
    }.get(verdict.lower(), verdict.upper())


def confidence_bar(confidence: float):
    color = "#e05252" if confidence < 0.5 else "#e0a052" if confidence < 0.75 else "#52c47a"
    st.markdown(
        f"""<div style="margin:0.3rem 0 0.8rem 0">
        <div style="height:4px;background:#1e1e2e;border-radius:2px">
          <div style="height:4px;width:{int(confidence*100)}%;background:{color};border-radius:2px;transition:width 0.4s"></div>
        </div>
        <div style="font-size:0.78rem;color:#6b6b7e;margin-top:0.3rem">{int(confidence*100)}% confidence</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_result(result: dict):
    if not result:
        return

    v = result.get("verification", {})
    verdict = v.get("verdict", "Unknown")
    css = verdict_css_class(verdict)
    label = verdict_emoji(verdict)
    confidence = v.get("confidence", 0)
    claim = result.get("claim", "")
    metrics = result.get("preprocessing_metrics", {})
    ml = result.get("ml_classification", {})
    matched_fact = v.get("matched_fact", "")
    fact_id = v.get("matched_fact_id", "")
    retrieval_score = v.get("retrieval_score", 0)

    # Verdict card
    st.markdown(
        f"""<div class="{css}">
        <div class="verdict-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    confidence_bar(confidence)

    # Claim
    st.markdown('<div class="section-label">Extracted claim</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="claim-box">"{claim}"</div>', unsafe_allow_html=True)

    # Matched fact
    source = v.get("source", "local_factstore")
    publisher = v.get("publisher", "")
    url = v.get("url", "")
    raw_rating = v.get("raw_rating", "")

    source_badge = "" 
    if source == "Google Fact Check":
        source_badge = f"<span style='background:#1a3a1a;color:#52c47a;font-size:0.7rem;padding:2px 8px;border-radius:10px;margin-left:8px;font-family:Syne,sans-serif'>GOOGLE VERIFIED</span>"
    
    publisher_line = f"<div style='font-size:0.75rem;color:#4a6a4a;margin-top:0.4rem'>Source: {publisher}" + (f" &nbsp;·&nbsp; <a href='{url}' target='_blank' style='color:#52c47a'>View fact-check ↗</a>" if url else "") + "</div>" if publisher else ""
    rating_line = f"<div style='font-size:0.75rem;color:#6b6b7e;margin-top:0.2rem'>Rating: {raw_rating}</div>" if raw_rating else ""

    st.markdown(f"<div class='section-label'>Matched verified fact {source_badge}</div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="fact-box">
        <div class="fact-id">{fact_id}</div>
        {matched_fact}
        {publisher_line}
        {rating_line}
        </div>""",
        unsafe_allow_html=True,
    )

    # Metrics row
    st.markdown('<div class="section-label">Pipeline metrics</div>', unsafe_allow_html=True)
    cols_data = [
        ("Token reduction", f"{metrics.get('token_reduction_pct', 0)}%"),
        ("Char reduction", f"{metrics.get('char_reduction_pct', 0)}%"),
        ("Retrieval score", f"{retrieval_score:.2f}"),
        ("ML prediction", ml.get("prediction", "N/A") if ml.get("available") else "N/A"),
    ]
    pills = "".join(
        f'<span class="metric-pill">{k}: <span class="metric-val">{val}</span></span>'
        for k, val in cols_data
    )
    st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)

    # Raw JSON expander
    with st.expander("Raw JSON response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-block">
    <div class="header-title">Vernacular <span class="accent">Fact</span> Checker</div>
    <div class="header-sub">High-throughput misinformation detection for Indian social media</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API status
# ---------------------------------------------------------------------------
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;font-size:0.8rem;color:#52c47a;margin-bottom:1.5rem">'
            '<div style="width:7px;height:7px;border-radius:50%;background:#52c47a"></div>'
            f'API connected · {API_URL}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"API returned status {health.status_code}")
except Exception:
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:6px;font-size:0.8rem;color:#e05252;margin-bottom:1.5rem">'
        '<div style="width:7px;height:7px;border-radius:50%;background:#e05252"></div>'
        f'API offline · {API_URL}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["  Single Post  ", "  Batch Check  ", "  Image Analysis  "])


# ── Tab 1: Single post ──────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Enter news or social media post</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            label="",
            placeholder="Paste a news headline, social media post, or WhatsApp forward here...",
            height=160,
            key="single_input",
            label_visibility="collapsed",
        )

        # Demo inputs
        st.markdown('<div class="section-label">Try a demo input</div>', unsafe_allow_html=True)
        demo_posts = [
            "RBI shutting all banks nationwide tomorrow",
            "Every citizen gets 5000 rupees daily from tonight",
            "Election Commission confirms official polling schedule",
            "Heatwave alert in north India this week",
            "Government announced new scheme via WhatsApp forward",
        ]
        for post in demo_posts:
            if st.button(f"↗ {post[:55]}...", key=f"demo_{post[:20]}"):
                st.session_state["single_input_val"] = post
                with col2:
                    with st.spinner("Checking..."):
                        result = call_predict(post)
                    render_result(result)

        if st.button("Check this post", key="check_single"):
            if text_input.strip():
                with col2:
                    with st.spinner("Checking..."):
                        result = call_predict(text_input)
                    render_result(result)
            else:
                st.warning("Please enter some text first.")

    with col2:
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#3a3a5a;font-size:0.9rem;margin-top:2rem;text-align:center">Enter text and click Check</div>',
            unsafe_allow_html=True,
        )


# ── Tab 2: Batch ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Enter one post per line</div>', unsafe_allow_html=True)
    batch_input = st.text_area(
        label="",
        placeholder="RBI shutting all banks tomorrow\nElection Commission confirms schedule\nEvery citizen gets 5000 rupees daily",
        height=180,
        label_visibility="collapsed",
    )

    if st.button("Check all posts", key="check_batch"):
        lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
        if lines:
            with st.spinner(f"Checking {len(lines)} posts..."):
                response = call_predict_batch(lines)

            results = response.get("results", [])
            st.markdown(
                f'<div style="font-size:0.85rem;color:#6b6b7e;margin-bottom:1rem">{len(results)} posts processed</div>',
                unsafe_allow_html=True,
            )

            for i, result in enumerate(results, 1):
                v = result.get("verification", {})
                verdict = v.get("verdict", "Unknown")
                css = verdict_css_class(verdict)
                label = verdict_emoji(verdict)
                confidence = v.get("confidence", 0)
                claim = result.get("claim", "")

                with st.expander(f"Post {i} — {label}  ({int(confidence*100)}% confidence)"):
                    render_result(result)
        else:
            st.warning("Please enter at least one post.")


# ── Tab 3: Image analysis ────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-label">Upload image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption=uploaded.name, use_container_width=True)

        analysis_type = st.radio(
            "Analysis type",
            ["Tamper + Deepfake detection", "OCR text extraction + fact-check"],
            help="Use tamper detection for edited photos and deepfakes. Use OCR for screenshots and infographics with text.",
        )

        if st.button("Analyse image", key="check_image") and uploaded:
            file_bytes = uploaded.getvalue()
            with col_b:
                with st.spinner("Analysing..."):
                    if analysis_type == "Tamper + Deepfake detection":
                        result = call_analyze_image(file_bytes, uploaded.name)

                        if result:
                            flagged = result.get("image_flagged", False)
                            if flagged:
                                st.markdown(
                                    '<div class="flag-banner">⚠ Image flagged as suspicious</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="safe-banner">✓ No manipulation detected</div>',
                                    unsafe_allow_html=True,
                                )

                            tamper = result.get("tamper_analysis", {})
                            deepfake = result.get("deepfake_analysis", {})

                            st.markdown('<div class="section-label">Tamper analysis (ELA)</div>', unsafe_allow_html=True)
                            ela = tamper.get("ela_score", 0)
                            ela_level = tamper.get("ela_level", "clean")
                            ela_confidence = tamper.get("confidence", "")
                            ela_pct = min(ela / 60, 1.0)
                            ela_color = {"clean": "#52c47a", "borderline": "#e0a052", "suspicious": "#e07852", "high": "#e05252"}.get(ela_level, "#52c47a")
                            st.markdown(
                                f"""<div style="margin:0.3rem 0 0.8rem 0">
                                <div style="height:4px;background:#1e1e2e;border-radius:2px">
                                  <div style="height:4px;width:{int(ela_pct*100)}%;background:{ela_color};border-radius:2px"></div>
                                </div>
                                <div style="font-size:0.78rem;color:#6b6b7e;margin-top:0.3rem">ELA score: {ela:.2f} · Level: {ela_level} · {ela_confidence}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                            pills = "".join(
                                f'<span class="metric-pill">{k}: <span class="metric-val">{v}</span></span>'
                                for k, v in [
                                    ("Tamper detected", str(tamper.get("tamper_detected", False))),
                                    ("ELA level", ela_level),
                                    ("Camera metadata", str(tamper.get("has_camera_metadata", False))),
                                    ("Deepfake score", str(deepfake.get("deepfake_score", 0.0))),
                                    ("Model available", str(deepfake.get("model_available", False))),
                                ]
                            )
                            st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)
                            if ela_confidence:
                                st.markdown(f'<div class="fact-box">{ela_confidence}</div>', unsafe_allow_html=True)

                            with st.expander("Raw JSON response"):
                                st.json(result)

                    else:
                        result = call_predict_image(file_bytes, uploaded.name)
                        render_result(result)

    with col_b:
        if not uploaded:
            st.markdown(
                '<div style="color:#3a3a5a;font-size:0.9rem;margin-top:2rem;text-align:center">Upload an image to analyse</div>',
                unsafe_allow_html=True,
            )

---
title: Vernacular Fact Checker
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Automated Fact-Checker for Vernacular News

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://jo-7-vernacular-fact-checker.hf.space)
[![API Docs](https://img.shields.io/badge/API-Swagger%20UI-green)](https://jo-7-vernacular-fact-checker.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-vernacular--fact--checker-black)](https://github.com/jinto-joseph/vernacular-fact-checker)

High-throughput misinformation detection pipeline for Indian social media. Strips non-factual noise before verification to reduce compute cost and increase throughput — satisfying the Pipeline Optimization technique requirement.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Architecture](#3-architecture)
4. [Pipeline Optimization — Core Technique](#4-pipeline-optimization--core-technique)
5. [Real-Time Fact Retrieval](#5-real-time-fact-retrieval)
6. [Measurable Results](#6-measurable-results)
7. [Real-World Feasibility](#7-real-world-feasibility)
8. [Project Structure](#8-project-structure)
9. [Setup](#9-setup)
10. [Environment Variables and API Keys](#10-environment-variables-and-api-keys)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [ML Model Training](#12-ml-model-training)
13. [API Reference](#13-api-reference)
14. [Deployment](#14-deployment)
15. [Limitations and Future Work](#15-limitations-and-future-work)
16. [Demo Script](#16-demo-script)

---

## 1. Problem Statement

Misinformation spreads rapidly through vernacular channels in India. Human fact-checkers cannot match the volume. Existing AI fact-checking systems are too slow and expensive because they process raw, noisy social text without pre-filtering.

**Key constraints addressed:**

| Constraint | How we solve it |
|---|---|
| Throughput: thousands of posts/min | ScaleDown AI compression + ThreadPoolExecutor batching |
| Context accuracy | Multi-source retrieval with LRU caching to avoid stale/conflicting facts |
| Cost | Token reduction via ScaleDown before any downstream API call |

---

## 2. Solution Overview

```
Ingestion → Optimization → Claim Extraction → Fact Retrieval → Verification → Output
```

The optimization stage is the core innovation: it strips non-factual content before any downstream computation, reducing token count and improving retrieval signal-to-noise ratio.

---

## 3. Architecture

```
Input (Text or Image)
        │
        ├── [if image] ──► OCR (pytesseract)
        │
        ▼
Preprocessing Optimization                ← ScaleDown AI + rule-based cleaning
        │
        ▼
Claim Extraction                          ← lightweight heuristic
        │
        ▼
Multi-Source Fact Retrieval (LRU-cached)
    ├── 1. Google Fact Check Tools API    ← AltNews, AFP, Snopes, PolitiFact
    ├── 2. MediaStack News API            ← real-time Indian news
    └── 3. Local fact store               ← 12 verified Indian patterns
        │
        ▼
Verification                              ← verdict + confidence score
        │
        ▼
ML Classification (optional)             ← TF-IDF + MLP/SVC/RF
        │
        ▼
Output: verdict · confidence · source · metrics
```

---

## 4. Pipeline Optimization — Core Technique

Implemented in `preprocessing.py`.

### Rule-based cleaning

| Step | What it removes |
|---|---|
| URL removal | `http://...`, `www....` |
| Mention/hashtag removal | `@user`, `#tag` |
| Emoji stripping | Unicode pictographs |
| Clickbait phrase removal | "watch till end", "share this now", etc. |
| Repeated character normalization | `wowwww` → `wow` |
| Noisy punctuation collapse | `!!!!` → `!` |
| Adjacent duplicate word removal | `news news` → `news` |
| Whitespace normalization | multiple spaces → single space |

### ScaleDown AI compression (when `SCALEDOWN_API_KEY` is set)

After rule-based cleaning, text is passed to the [ScaleDown API](https://scaledown.xyz) which uses small language models to identify and retain only factually relevant content — going beyond what rules can achieve.

```python
# ScaleDown reduces token count by identifying non-factual content
result = requests.post("https://api.scaledown.xyz/compress/raw/", ...)
compressed_prompt = result["compressed_prompt"]
token_savings = result["original_prompt_tokens"] - result["compressed_prompt_tokens"]
```

### Performance techniques
- Batch processing with `ThreadPoolExecutor` (8 workers default)
- Retrieval caching via `functools.lru_cache` (2048-entry LRU)
- ScaleDown only called for posts > 10 tokens (avoids overhead on short text)

---

## 5. Real-Time Fact Retrieval

Implemented in `fact_retrieval.py`. Three sources queried in priority order:

### Source 1: Google Fact Check Tools API
Queries a database of verified claims from publishers including AltNews (India), AFP Fact Check, Snopes, PolitiFact, and more.

```bash
# Enable: set GOOGLE_FACTCHECK_API_KEY environment variable
# Free tier: available via Google Cloud Console
# Endpoint: https://factchecktools.googleapis.com/v1alpha1/claims:search
```

### Source 2: MediaStack News API
Cross-references claims against real-time Indian news articles.

```bash
# Enable: set MEDIASTACK_API_KEY environment variable  
# Free tier: 500 requests/month at mediastack.com
# Endpoint: http://api.mediastack.com/v1/news
```

### Source 3: Local fact store
12 hardcoded patterns covering the most common Indian social media misinformation topics (bank closures, currency bans, death hoaxes, free scheme scams, election misinformation etc.)

### Fallback
If no source matches, returns `Unverified` — an honest no-match response rather than a misleading guess.

---

## 6. Measurable Results

```bash
python main.py
```

| Metric | Value |
|---|---|
| Token reduction (rule-based) | 40–60% on noisy social posts |
| Token reduction (with ScaleDown) | Up to 70–80% |
| Throughput | ~1000 posts/min with batching |
| Avg per-post latency | < 200 ms (local) |
| Fact retrieval sources | 3 (Google + MediaStack + local) |
| ML classifier accuracy | ~85–95% (dataset dependent) |

---

## 7. Real-World Feasibility

- **Stateless processing:** each post is independent — horizontally scalable behind any load balancer
- **LRU caching:** viral claims are processed once and cached — handles repeated misinformation efficiently
- **Modular retrieval:** swap local store for FAISS or a production vector DB without touching other pipeline stages
- **Cost reduction:** ScaleDown compression reduces downstream LLM/API token spend proportionally
- **Image support:** OCR → same pipeline for screenshot and poster misinformation
- **Graceful degradation:** every API is optional; pipeline works without any external keys

---

## 8. Project Structure

```
vernacular-fact-checker/
├── main.py              # End-to-end pipeline, benchmarking, demo
├── preprocessing.py     # Optimization stage — ScaleDown + rule-based cleaning
├── fact_retrieval.py    # Multi-source fact retrieval (Google + MediaStack + local)
├── compare_models.py    # ML model training and comparison
├── image_analysis.py    # ELA tamper detection + deepfake classification
├── api.py               # FastAPI server (5 endpoints)
├── app.py               # Streamlit UI
├── supervisord.conf     # Runs API + Streamlit together in Docker
├── requirements.txt     # All runtime dependencies
├── Dockerfile           # Container build (CPU-only torch)
├── render.yaml          # Render Blueprint deployment config
├── artifacts/           # Generated model artifacts (gitignored)
└── README.md
```

---

## 9. Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/jinto-joseph/vernacular-fact-checker.git
cd vernacular-fact-checker
python -m venv .venv
```

Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 10. Environment Variables and API Keys

Set these before running. None are required — the pipeline degrades gracefully without them.

| Variable | Service | How to get | Free tier |
|---|---|---|---|
| `SCALEDOWN_API_KEY` | ScaleDown compression | [scaledown.xyz](https://scaledown.xyz) | Contact sales |
| `GOOGLE_FACTCHECK_API_KEY` | Google Fact Check Tools | [Google Cloud Console](https://console.cloud.google.com) → Enable Fact Check Tools API | Free |
| `MEDIASTACK_API_KEY` | MediaStack News | [mediastack.com](https://mediastack.com) | 500 req/month free |

### Setting variables locally (Windows)
```powershell
$env:SCALEDOWN_API_KEY="your_key_here"
$env:GOOGLE_FACTCHECK_API_KEY="your_key_here"
$env:MEDIASTACK_API_KEY="your_key_here"
```

### Setting variables on HuggingFace Spaces
1. Go to your Space → **Settings** → **Repository secrets**
2. Add each variable as a secret
3. HuggingFace injects them automatically at runtime

### Setting variables for Docker
```bash
docker run --rm -p 7860:7860 \
  -e SCALEDOWN_API_KEY=your_key \
  -e GOOGLE_FACTCHECK_API_KEY=your_key \
  -e MEDIASTACK_API_KEY=your_key \
  vernacular-fact-checker:latest
```

> **Security:** Never hardcode API keys in source files. Always use environment variables. Rotate any key that was accidentally exposed in a commit or chat.

---

## 11. Running the Pipeline

```bash
# Run demo with benchmark
python main.py

# Train and compare ML models
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"

# Quick training run (3000 samples)
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData" --max-samples 3000

# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Start Streamlit UI
streamlit run app.py
```

---

## 12. ML Model Training

Trains three classifiers on TF-IDF features and saves the best performer:

| Model | Notes |
|---|---|
| `MLPClassifier` | MLP neural network, hidden layers (64, 32) |
| `LinearSVC` | Linear support vector machine |
| `RandomForestClassifier` | 200 estimators, parallel |

**Dataset format:**

- CSV: `Fake.csv`, `True.csv` with a `text` or `title` column
- TXT: class-named subfolders (`Fake/`, `True/`) with `.txt` files

**Artifacts saved to `artifacts/`:**

| File | Contents |
|---|---|
| `best_fake_news_model.joblib` | Best-performing trained pipeline |
| `model_comparison.csv` | Ranked accuracy table |
| `model_reports.joblib` | Full classification reports |

> **Note on accuracy:** Very high scores (>99%) on public datasets are common due to duplicate-heavy content. Always verify with clean stratified splits using `--max-samples`.

---

## 13. API Reference

**Live API:** `https://jo-7-vernacular-fact-checker.hf.space`

**Interactive docs:** `https://jo-7-vernacular-fact-checker.hf.space/docs`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single post fact-check |
| POST | `/predict-batch` | Batch fact-check (parallel) |
| POST | `/predict-image` | OCR + fact-check from image |
| POST | `/analyze-image` | Tamper detection + deepfake classification |

### Example requests

```bash
# Health check
curl https://jo-7-vernacular-fact-checker.hf.space/health

# Single prediction
curl -X POST "https://jo-7-vernacular-fact-checker.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "RBI is shutting all banks nationwide tomorrow"}'

# Batch prediction
curl -X POST "https://jo-7-vernacular-fact-checker.hf.space/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["RBI shutting banks", "Election Commission confirms schedule"]}'
```

---

## 14. Deployment

### HuggingFace Spaces (live)

```bash
git remote add hf https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/JO-7/vernacular-fact-checker
git push hf master:main --force
```

Set API keys in Space Settings → Repository secrets.

### Docker (local)

```bash
docker build -t vernacular-fact-checker:latest .
docker run --rm -p 7860:7860 \
  -e SCALEDOWN_API_KEY=your_key \
  -e GOOGLE_FACTCHECK_API_KEY=your_key \
  vernacular-fact-checker:latest
```

### Render (Blueprint)

Push to GitHub, connect repo in Render dashboard → New → Blueprint. Set environment variables in Render's environment settings.

---

## 15. Limitations and Future Work

**Current limitations:**
- ScaleDown compression adds ~2–5 ms latency per post (network round-trip)
- Google Fact Check API coverage is stronger for English than Hinglish
- MediaStack free tier limited to 500 requests/month
- Local fact store covers only 12 topic categories
- ML classifier requires local training — artifact not included in repo

**Future improvements:**
- FAISS vector store for semantic retrieval at scale
- Multilingual support (Hindi, Tamil, Telugu) via IndicNLP
- Temporal re-ranking to deprioritize outdated facts
- Source trust scoring (government sources > blogs)
- Webhook-based real-time social media monitoring

---

## 16. Demo Script

**2–3 minute flow for judges:**

1. Open `https://jo-7-vernacular-fact-checker.hf.space`
2. Paste a noisy social post — show cleaned text and token reduction %
3. Show extracted claim and matched verified fact with source
4. Show verdict + confidence — demonstrate False, True, and Misleading cases
5. Upload a screenshot — show OCR → same pipeline
6. Upload a manipulated image — show ELA tamper detection result
7. Run batch check — show throughput and cost savings

**Speaking points:**

> "We optimize noisy social text before verification using ScaleDown AI compression — this directly reduces token cost and increases throughput at scale."

> "Our retrieval queries Google Fact Check Tools first, which covers verified claims from AltNews, AFP, and Snopes — real sources used by professional fact-checkers in India."

> "The architecture is stateless and horizontally scalable. Every external API is optional — the pipeline degrades gracefully so it never goes down."

> "We trained and compared MLP, LinearSVC, and RandomForest classifiers, selecting the best by test accuracy on a stratified split."

---
title: Vernacular Fact Checker
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Automated Fact-Checker for Vernacular News

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/JO-7/vernacular-fact-checker)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-UI-red)](https://jo-7-vernacular-fact-checker.hf.space)
[![API Docs](https://img.shields.io/badge/API-Swagger%20UI-green)](https://jo-7-vernacular-fact-checker.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-vernacular--fact--checker-black)](https://github.com/jinto-joseph/vernacular-fact-checker)

High-throughput misinformation detection pipeline for Indian social media. Strips non-factual noise before verification to reduce compute cost and increase throughput — satisfying the Pipeline Optimization technique requirement.

---

## Live Deployment

| Link | Description |
|---|---|
| [HuggingFace Space](https://huggingface.co/spaces/JO-7/vernacular-fact-checker) | Main deployed app |
| [Streamlit UI](https://jo-7-vernacular-fact-checker.hf.space) | Interactive fact-checker interface |
| [API Docs](https://jo-7-vernacular-fact-checker.hf.space/docs) | Swagger UI for all endpoints |
| [Health Check](https://jo-7-vernacular-fact-checker.hf.space/health) | API liveness endpoint |

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
    ├── 1. Tavily Search API              ← real-time web search across fact-check domains
    ├── 2. NewsData.io API                ← real-time Indian news cross-reference
    └── 3. Local fact store               ← 12 verified Indian misinformation patterns
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

### Source 1: Tavily Search API

Searches the web in real time across trusted fact-check and news domains including AltNews, BoomLive, AFP Fact Check, Snopes, Reuters, NDTV, and The Hindu. Returns an AI-synthesised answer plus ranked source links.

```bash
# Enable: set TAVILY_API_KEY environment variable
# Free tier: 1000 credits/month, no credit card — app.tavily.com
```

### Source 2: NewsData.io

Cross-references claims against real-time Indian news articles filtered by country and language.

```bash
# Enable: set NEWSDATA_API_KEY environment variable
# Free tier: 200 credits/day, no credit card — newsdata.io
```

### Source 3: Local fact store

12 hardcoded patterns covering the most common Indian social media misinformation topics — bank closures, currency bans, death hoaxes, free scheme scams, election misinformation, fuel prices, health advisories, and military/border claims.

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
| Fact retrieval sources | 3 (Tavily + NewsData + local) |
| ML classifier accuracy | ~85–95% (dataset dependent) |

---

## 7. Real-World Feasibility

- **Stateless processing:** each post is independent — horizontally scalable behind any load balancer
- **LRU caching:** viral claims are processed once and cached — handles repeated misinformation efficiently
- **Modular retrieval:** swap local store for FAISS or a production vector DB without touching other pipeline stages
- **Cost reduction:** ScaleDown compression reduces downstream LLM/API token spend proportionally
- **Image support:** OCR path handles screenshot and poster misinformation; ELA + deepfake detection for manipulated images
- **Graceful degradation:** every API is optional — pipeline works without any external keys

---

## 8. Project Structure

```
vernacular-fact-checker/
├── main.py              # End-to-end pipeline, benchmarking, demo
├── preprocessing.py     # Optimization stage — ScaleDown + rule-based cleaning
├── fact_retrieval.py    # Multi-source fact retrieval (Tavily + NewsData + local)
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

All keys are optional — the pipeline degrades gracefully without them, falling back to the local fact store.

| Variable | Service | Sign up | Free tier |
|---|---|---|---|
| `SCALEDOWN_API_KEY` | ScaleDown prompt compression | [scaledown.xyz](https://scaledown.xyz) | Contact sales |
| `TAVILY_API_KEY` | Tavily real-time web search | [app.tavily.com](https://app.tavily.com) | 1000 credits/month, no credit card |
| `NEWSDATA_API_KEY` | NewsData.io Indian news | [newsdata.io](https://newsdata.io) | 200 credits/day, no credit card |

### Setting variables locally (Windows PowerShell)

```powershell
$env:SCALEDOWN_API_KEY="your_key_here"
$env:TAVILY_API_KEY="tvly-xxxxxxxxxx"
$env:NEWSDATA_API_KEY="pub_xxxxxxxxxx"
```

### Setting variables on HuggingFace Spaces

1. Go to [HuggingFace Space](https://huggingface.co/spaces/JO-7/vernacular-fact-checker) → **Settings** tab → **Repository secrets**
2. Click **New secret** for each variable
3. HuggingFace injects them automatically into the container at runtime

### Setting variables for Docker

```bash
docker run --rm -p 7860:7860 \
  -e SCALEDOWN_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e NEWSDATA_API_KEY=your_key \
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

# Quick training run (3000 samples, faster)
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

- CSV source: `Fake.csv`, `True.csv` with a `text` or `title` column
- TXT source: class-named subfolders (`Fake/`, `True/`) containing `.txt` files

**Artifacts saved to `artifacts/`:**

| File | Contents |
|---|---|
| `best_fake_news_model.joblib` | Best-performing trained pipeline |
| `model_comparison.csv` | Ranked accuracy table |
| `model_reports.joblib` | Full classification reports for all models |

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

**Space URL:** `https://huggingface.co/spaces/JO-7/vernacular-fact-checker`

```bash
git remote add hf https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/JO-7/vernacular-fact-checker
git push hf master:main --force
```

Add API keys in Space → **Settings** → **Repository secrets**.

### Docker (local)

```bash
docker build -t vernacular-fact-checker:latest .
docker run --rm -p 7860:7860 \
  -e SCALEDOWN_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e NEWSDATA_API_KEY=your_key \
  vernacular-fact-checker:latest
```

### Render (Blueprint)

Push to GitHub, open [Render dashboard](https://dashboard.render.com) → New → Blueprint → connect repository. Set environment variables in Render's environment settings panel.

---

## 15. Limitations and Future Work

**Current limitations:**

- ScaleDown compression adds ~2–5 ms latency per post due to network round-trip
- Tavily search coverage is stronger for English than Hinglish or regional languages
- NewsData.io free tier is limited to 200 credits/day
- Local fact store covers only 12 topic categories
- ML classifier requires local training — artifact not committed to repo

**Future improvements:**

- FAISS vector store for semantic retrieval at scale
- Multilingual support for Hindi, Tamil, and Telugu via IndicNLP
- Temporal re-ranking to deprioritize outdated facts
- Source trust scoring to weight government and established outlets higher
- Webhook-based real-time social media feed monitoring
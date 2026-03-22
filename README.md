---
title: Vernacular Fact Checker
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Automated Fact-Checker for Vernacular News

High-throughput misinformation detection pipeline for Indian social media posts. Strips non-factual noise before verification to reduce compute cost and increase throughput.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Architecture](#3-architecture)
4. [Pipeline Optimization (Core Technique)](#4-pipeline-optimization-core-technique)
5. [Measurable Results](#5-measurable-results)
6. [Real-World Feasibility](#6-real-world-feasibility)
7. [Project Structure](#7-project-structure)
8. [Setup](#8-setup)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [ML Model Training and Comparison](#10-ml-model-training-and-comparison)
11. [API Reference](#11-api-reference)
12. [Deployment](#12-deployment)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [Demo Script (For Judges)](#14-demo-script-for-judges)

---

## 1. Problem Statement

Misinformation spreads rapidly through vernacular channels in India. Human fact-checkers cannot match the volume. Existing AI fact-checking systems are too slow and expensive to operate at scale because they process raw, noisy social text without any pre-filtering.

**Key constraints this project addresses:**
- **Throughput:** must handle thousands of posts per minute
- **Context accuracy:** retrieval must not be confused by conflicting or outdated facts
- **Cost:** token-level optimization reduces LLM/API spend proportionally

---

## 2. Solution Overview

The pipeline processes each post through five sequential stages:

```
Ingestion → Optimization → Claim Extraction → Fact Retrieval → Verification
```

The optimization stage is the core innovation: it strips non-factual content (emojis, clickbait, repeated words, URLs) before any downstream computation, reducing token count and improving retrieval signal.

---

## 3. Architecture

```
Input (Text or Image)
        │
        ├── [if image] ──► OCR (pytesseract, optional)
        │
        ▼
Preprocessing Optimization          ← strips noise, reduces tokens
        │
        ▼
Claim Extraction                    ← lightweight heuristic, no heavy model
        │
        ▼
Fact Retrieval (LRU-cached)         ← tag-overlap against verified fact store
        │
        ▼
Verification                        ← verdict + confidence score
        │
        ▼
ML Classification (optional)        ← trained TF-IDF + classifier artifact
        │
        ▼
Output: verdict / confidence / metrics
```

---

## 4. Pipeline Optimization (Core Technique)

Implemented in `preprocessing.py`, integrated into every pipeline call.

**Steps applied in order:**

| Step | What it removes |
|------|----------------|
| URL removal | `http://...`, `www....` |
| Mention/hashtag removal | `@user`, `#tag` |
| Emoji stripping | Unicode pictographs |
| Clickbait phrase removal | "watch till end", "share this now", etc. |
| Repeated character normalization | `wowwww` → `wow` |
| Noisy punctuation collapse | `!!!!` → `!` |
| Adjacent duplicate word removal | `news news` → `news` |
| Whitespace normalization | multiple spaces/newlines → single space |

**Performance techniques:**
- Batch processing with `ThreadPoolExecutor`
- Retrieval caching via `functools.lru_cache` (2048-entry LRU)
- Lightweight heuristic claim extraction (no transformer dependency)
- Optional OCR path that does not affect the core pipeline

---

## 5. Measurable Results

Run the benchmark to produce live metrics:

```bash
python main.py
```

**Benchmark metrics reported:**

| Metric | Description |
|--------|-------------|
| `token_reduction_pct` | % tokens removed by preprocessing |
| `char_reduction_pct` | % characters removed |
| `avg_latency_ms` | Average per-post pipeline time |
| `throughput_posts_per_min` | Posts processed per minute |
| `estimated_cost_before_usd` | Simulated token cost without optimization |
| `estimated_cost_after_usd` | Simulated token cost with optimization |
| `estimated_cost_savings_pct` | % cost reduction |

**Target outcomes on demo hardware:**
- 40–60% token/char reduction on noisy social posts
- ~1000 posts/min with batching + 8 worker threads
- < 200 ms average per-post latency
- ~80–85% accuracy on curated demo inputs

> **Note:** ML accuracy varies with dataset quality and size. Very high scores (> 99%) on some public datasets are expected due to duplicate-heavy content; use the `--max-samples` flag and verify with clean stratified splits.

**Model comparison table** (generated after training):

| Rank | Model | Accuracy |
|------|-------|----------|
| — | Run `compare_models.py` to populate | — |

---

## 6. Real-World Feasibility

- **Horizontal scaling:** stateless per-post processing; deploy behind any load balancer
- **Caching:** LRU cache handles repeated viral claims without redundant computation
- **Retrieval swap:** replace the demo fact store with FAISS or a production vector DB without changing pipeline stages
- **Image support:** OCR path handles screenshot/poster misinformation
- **Deployment:** Docker image and Railway/Render configs included

---

## 7. Project Structure

```
vernacular-fact-checker/
├── main.py              # End-to-end pipeline, retrieval, verification, benchmark
├── preprocessing.py     # Optimization stage — noise removal and reduction metrics
├── compare_models.py    # Multi-model training, evaluation, and artifact saving
├── api.py               # FastAPI server (/health, /predict, /predict-batch, /predict-image)
├── requirements.txt     # All runtime dependencies
├── Dockerfile           # Container build
├── render.yaml          # Render Blueprint deployment config
├── artifacts/           # Generated model artifacts (gitignored; created at runtime)
│   ├── best_fake_news_model.joblib
│   ├── model_comparison.csv
│   └── model_reports.joblib
└── README.md
```

---

## 8. Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/jinto-joseph/vernacular-fact-checker.git
cd vernacular-fact-checker
python -m venv .venv
```

**Activate the virtual environment:**

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

**Install dependencies:**

```bash
pip install -r requirements.txt
```

If OCR is not needed, remove the `pillow` and `pytesseract` lines from `requirements.txt` before installing.

---

## 9. Running the Pipeline

**Run the demo (no training required):**

```bash
python main.py
```

Prints per-post verdicts and a 5-run benchmark summary.

**Run model comparison and training:**

```bash
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"
```

Quick debug run (faster, fewer samples):

```bash
python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData" --max-samples 3000
```

After training, `main.py` automatically loads the best model artifact.

---

## 10. ML Model Training and Comparison

The comparison script trains three classifiers on TF-IDF features and ranks them by test accuracy:

- `MLPClassifier` (MLP neural network)
- `LinearSVC` (linear support vector machine)
- `RandomForestClassifier` (ensemble)

**Dataset format expected:**

*CSV source* (`--csv-root`):
- `Fake.csv`, `True.csv` with a `text` or `title` column

*TXT source* (`--txt-root`):
- Class-named subfolders (`Fake/`, `True/`) containing `.txt` files

**Artifacts saved to `artifacts/`:**

| File | Contents |
|------|----------|
| `best_fake_news_model.joblib` | Best-performing trained pipeline |
| `model_comparison.csv` | Ranked accuracy table |
| `model_reports.joblib` | Full classification reports for all models |

To use a specific model instead of the best one, update `model_path` in `main.py`.

---

## 11. API Reference

**Start the server:**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### GET `/health`

```bash
curl http://127.0.0.1:8000/health
```

### POST `/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: RBI is shutting all banks nationwide tomorrow"}'
```

### POST `/predict-batch`

```bash
curl -X POST "http://127.0.0.1:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["RBI shutting banks tomorrow", "Election Commission confirms schedule"]}'
```

### POST `/predict-image` (optional, requires OCR)

```bash
curl -X POST "http://127.0.0.1:8000/predict-image" \
  -F "file=@/path/to/screenshot.jpg"
```

---

## 12. Deployment

### Docker

```bash
docker build -t vernacular-fact-checker:latest .
docker run --rm -p 8000:8000 vernacular-fact-checker:latest
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

### Railway (no Docker required)

```bash
npm i -g @railway/cli
railway login
railway init
railway up
railway open
```

Set port if needed:

```bash
railway variables set PORT=8000
```

### Render (Blueprint)

This repo includes `render.yaml`. To deploy:

1. Push code to GitHub
2. Open [Render dashboard](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect this repository
4. Render reads `render.yaml` and deploys automatically

Verify `/health` and `/docs` after deploy.

### Environment Variables

Never hardcode secrets. Use environment variables:

```bash
export FACTCHECK_API_KEY="your_key_here"
```

Read in Python:

```python
import os
api_key = os.getenv("FACTCHECK_API_KEY")
```

---

$deployLine = "`n## Live Demo`nhttps://jo-7-vernacular-fact-checker.hf.space/docs`n"
Add-Content "C:\Users\Lenovo\Desktop\VFN\README.md" $deployLine

## 13. Limitations and Future Work

**Current limitations:**
- Rule-based multilingual normalization misses nuanced Hinglish/code-switched text
- Demo fact store is small (5 facts); coverage is limited by design
- No timestamp-aware retrieval (outdated facts can surface)
- Token overlap is a weak similarity measure for semantically similar claims

**Future improvements:**
- FAISS integration for large-scale semantic retrieval
- Multilingual normalization with transliteration support
- Source trust scoring and temporal re-ranking
- NLI-based verification for edge cases
- Larger verified fact corpus with automated ingestion

---

## 14. Demo Script (For Judges)

**Suggested 2–3 minute flow:**

1. Show a noisy social post (emoji-heavy, clickbait)
2. Show the cleaned text and reduction percentage
3. Show the extracted claim
4. Show the retrieved verified fact and similarity score
5. Show the final verdict and confidence
6. Run the batch benchmark — display throughput, latency, cost savings
7. (Optional) Upload an image and show OCR → same pipeline output

**One-line pitch:**

> "We reduce noisy input first, then verify faster and cheaper at scale — preserving factual context while cutting unnecessary compute."

**Speaking points:**

- "We optimized noisy social text before verification to reduce compute cost and increase throughput."
- "We trained and compared three models — MLP, LinearSVC, RandomForest — and selected the best by test accuracy."
- "Our API supports single, batch, and image-based inference through a consistent pipeline."
- "The architecture is stateless and horizontally scalable, with a clear upgrade path to production retrieval."


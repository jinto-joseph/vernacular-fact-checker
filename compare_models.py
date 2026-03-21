"""
compare_models.py
-----------------
Train and compare MLP, LinearSVC, and RandomForest classifiers on the
fake-news dataset, then save artifacts for the best model.

Datasets expected:
    --csv-root  Folder with Fake.csv / True.csv  (e.g. "News _dataset")
    --txt-root  Folder with class-wise .txt files (e.g. "FakeNewsData")

Run:
    python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData"

Quick debug run (faster):
    python compare_models.py --csv-root "News _dataset" --txt-root "FakeNewsData" --max-samples 3000

Artifacts saved to --save-dir (default: artifacts/):
    model_comparison.csv
    model_reports.joblib
    best_fake_news_model.joblib
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@dataclass
class Record:
    text: str
    label: str
    source: str


def _normalize_label(raw_label: str) -> str:
    """Map raw file/folder name to a canonical label: Fake, True, or Unknown."""
    value = raw_label.lower()
    if any(kw in value for kw in ("fake", "satire", "false")):
        return "Fake"
    if any(kw in value for kw in ("true", "real", "genuine")):
        return "True"
    return "Unknown"


def _load_csv_source(root: Path) -> List[Record]:
    """
    Load labeled records from CSV files.

    Each CSV filename determines the label (e.g. Fake.csv -> "Fake").
    Requires a 'text' or 'title' column.
    """
    records: List[Record] = []
    if not root.exists():
        return records

    for csv_path in root.glob("*.csv"):
        label = _normalize_label(csv_path.stem)
        if label == "Unknown":
            continue

        df = pd.read_csv(csv_path)
        text_col = next(
            (col for col in ("text", "title") if col in df.columns), None
        )
        if text_col is None:
            continue

        for raw_text in df[text_col].dropna().astype(str):
            cleaned = clean_text(raw_text)
            if len(cleaned.split()) < 4:
                continue
            records.append(Record(text=cleaned, label=label, source=csv_path.name))

    return records


def _load_txt_source(root: Path) -> List[Record]:
    """
    Load labeled records from .txt files organized in class-named folders.

    Folder name determines the label (e.g. Fake/ -> "Fake").
    """
    records: List[Record] = []
    if not root.exists():
        return records

    for path in root.rglob("*.txt"):
        label = _normalize_label(str(path.parent.name))
        if label == "Unknown":
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        cleaned = clean_text(content)
        if len(cleaned.split()) < 4:
            continue

        records.append(
            Record(
                text=cleaned,
                label=label,
                source=str(path.relative_to(root)),
            )
        )

    return records


def load_combined_dataset(
    csv_root: str,
    txt_root: str,
    max_samples: int = 0,
) -> pd.DataFrame:
    """
    Merge CSV and TXT sources into a single, deduplicated DataFrame.

    Args:
        csv_root:    Path to CSV dataset folder.
        txt_root:    Path to TXT dataset folder.
        max_samples: Cap on total rows (0 = no cap). Applied after dedup.

    Returns:
        DataFrame with columns: text, label, source.

    Raises:
        ValueError: If no valid records are found in either source.
    """
    all_records = _load_csv_source(Path(csv_root)) + _load_txt_source(Path(txt_root))
    df = pd.DataFrame([r.__dict__ for r in all_records])

    if df.empty:
        raise ValueError(
            "No training records found. "
            "Check that --csv-root and --txt-root point to the correct folders."
        )

    df = (
        df[df["label"].isin(["Fake", "True"])]
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def build_model_zoo() -> Dict[str, object]:
    """Return the set of classifiers to compare."""
    return {
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=220,
            random_state=42,
        ),
        "LinearSVC": LinearSVC(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }


def _build_tfidf() -> TfidfVectorizer:
    """Shared TF-IDF configuration used across all model pipelines."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=40_000,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, object]]:
    """
    Train all models on a stratified split and evaluate accuracy.

    Args:
        df:        DataFrame with 'text' and 'label' columns.
        test_size: Fraction of data used for evaluation.

    Returns:
        Tuple of:
            - ranked_df: DataFrame ranked by accuracy (descending).
            - reports:   Per-model classification_report dicts.
            - pipelines: Trained sklearn Pipeline objects keyed by model name.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )

    rows: List[dict] = []
    reports: Dict[str, dict] = {}
    trained_pipelines: Dict[str, object] = {}

    for model_name, clf in build_model_zoo().items():
        pipeline = Pipeline(
            [("tfidf", _build_tfidf()), ("clf", clf)]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        rows.append({"Model": model_name, "Accuracy": round(float(acc) * 100.0, 4)})
        reports[model_name] = classification_report(y_test, preds, output_dict=True)
        trained_pipelines[model_name] = pipeline

    ranked = (
        pd.DataFrame(rows)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    return ranked, reports, trained_pipelines


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_ranked_table(ranked_df: pd.DataFrame) -> None:
    """Pretty-print the model comparison table to stdout."""
    print("\n=== Model Comparison (Ranked by Accuracy) ===")
    header = f"{'Rank':<6}{'Model':<28}{'Accuracy':>12}"
    print(header)
    print("-" * len(header))
    for rank, row in ranked_df.iterrows():
        print(f"{rank:<6}{row['Model']:<28}{row['Accuracy']:>11.4f}%")


def save_artifacts(
    ranked_df: pd.DataFrame,
    reports: Dict[str, dict],
    pipelines: Dict[str, object],
    save_dir: str,
) -> None:
    """
    Persist comparison table, all reports, and the best-performing model.

    Args:
        ranked_df: Ranked accuracy DataFrame.
        reports:   Per-model classification report dicts.
        pipelines: Trained sklearn Pipeline objects.
        save_dir:  Output directory (created if absent).
    """
    os.makedirs(save_dir, exist_ok=True)
    base = Path(save_dir)

    ranked_path = base / "model_comparison.csv"
    report_path = base / "model_reports.joblib"
    best_name = ranked_df.iloc[0]["Model"]
    best_model_path = base / "best_fake_news_model.joblib"

    ranked_df.to_csv(ranked_path)
    joblib.dump(reports, report_path)
    joblib.dump(pipelines[best_name], best_model_path)

    print("\n=== Saved Artifacts ===")
    print(f"  Ranked table : {ranked_path}")
    print(f"  All reports  : {report_path}")
    print(f"  Best model   : {best_model_path}  ({best_name})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MLP vs LinearSVC vs RandomForest for fake-news classification."
    )
    parser.add_argument(
        "--csv-root",
        default="News _dataset",
        help="Folder containing Fake.csv / True.csv  (default: 'News _dataset')",
    )
    parser.add_argument(
        "--txt-root",
        default="FakeNewsData",
        help="Folder containing class-wise .txt files  (default: 'FakeNewsData')",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for evaluation  (default: 0.2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20_000,
        help="Cap on total training samples; 0 = no cap  (default: 20000)",
    )
    parser.add_argument(
        "--save-dir",
        default="artifacts",
        help="Directory for output artifacts  (default: artifacts/)",
    )
    args = parser.parse_args()

    # Load data
    df = load_combined_dataset(args.csv_root, args.txt_root, args.max_samples)

    print("=== Dataset Summary ===")
    print(f"  Total samples : {len(df)}")
    print("  Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count}")

    # Train and evaluate
    ranked, reports, pipelines = evaluate_models(df, test_size=args.test_size)
    print_ranked_table(ranked)

    # Persist artifacts
    save_artifacts(ranked, reports, pipelines, args.save_dir)


if __name__ == "__main__":
    run()

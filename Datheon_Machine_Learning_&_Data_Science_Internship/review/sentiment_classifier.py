"""TF-IDF sentiment classification for the local customer-review dataset."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
TEXT_COLUMN = "review_text"
TARGET_COLUMN = "sentiment"
ID_COLUMN = "review_id"
RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_FEATURES = 5_000
SUPPORTED_LABELS = {"positive", "negative", "neutral"}
NEGATION_WORDS = {"no", "nor", "not", "never"}
STOP_WORDS = ENGLISH_STOP_WORDS.difference(NEGATION_WORDS)
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Strip markup, lowercase, tokenize, remove stopwords, and lemmatize."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")

    text = html.unescape(re.sub(r"<[^>]*>", " ", text)).lower()
    tokens = re.findall(r"[a-z]+", text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    try:
        return " ".join(_LEMMATIZER.lemmatize(token) for token in tokens)
    except LookupError as error:
        raise LookupError(
            "NLTK WordNet data is missing. Install it with "
            "`python -c \"import nltk; nltk.download('wordnet')\"`."
        ) from error


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the review CSV, retaining neutral rows for reporting."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Review dataset was not found: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = {ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no reviews.")
    if frame[[TEXT_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Review text and sentiment labels must not be missing.")

    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    if (frame[TEXT_COLUMN] == "").any():
        raise ValueError("Review text must not be empty.")
    unexpected_labels = sorted(set(frame[TARGET_COLUMN]) - SUPPORTED_LABELS)
    if unexpected_labels:
        raise ValueError(f"Unsupported sentiment labels: {unexpected_labels}")
    return frame


def prepare_binary_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, int]:
    """Return positive/negative reviews and the number of excluded neutral rows."""
    binary_frame = frame.loc[frame[TARGET_COLUMN] != "neutral"]
    neutral_count = int((frame[TARGET_COLUMN] == "neutral").sum())
    counts = binary_frame[TARGET_COLUMN].value_counts()
    missing_classes = sorted({"positive", "negative"} - set(counts.index))
    if missing_classes or (counts < 2).any():
        raise ValueError(
            "Binary training requires at least two examples of both positive "
            "and negative sentiment."
        )
    return binary_frame[TEXT_COLUMN], binary_frame[TARGET_COLUMN], neutral_count


def build_models() -> dict[str, Pipeline]:
    """Create TF-IDF pipelines for Naive Bayes and Logistic Regression."""
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
        max_features=MAX_FEATURES,
    )
    return {
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", vectorizer), ("classifier", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=preprocess_text,
                        lowercase=False,
                        tokenizer=str.split,
                        token_pattern=None,
                        max_features=MAX_FEATURES,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def split_dataset(
    texts: pd.Series,
    labels: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Make a reproducible stratified split, keeping both classes in each set."""
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def evaluate_models(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit the model pipelines and calculate holdout metrics."""
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, float | str]] = []

    for model_name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=["negative", "positive"],
            average="binary",
            pos_label="positive",
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_positive": float(precision),
            "recall_positive": float(recall),
            "f1_positive": float(f1),
        }
        results[model_name] = {
            "estimator": estimator,
            "predictions": predictions,
            "metrics": metrics,
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=["negative", "positive"],
                output_dict=True,
                zero_division=0,
            ),
        }
        summary_rows.append({"model": model_name, **metrics})

    summary = pd.DataFrame(summary_rows).sort_values(
        "f1_positive", ascending=False, ignore_index=True
    )
    return results, summary


def predict_sentiment(text: str, model: Pipeline) -> dict[str, str | float]:
    """Predict sentiment and confidence for one custom review string."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty review.")
    prediction = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(prediction)
    return {
        "sentiment": prediction,
        "confidence": float(probabilities[class_index]),
    }


def run_benchmark(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate both models, saving metrics and run metadata."""
    frame = load_dataset(data_path)
    texts, labels, neutral_count = prepare_binary_dataset(frame)
    X_train, X_test, y_train, y_test = split_dataset(texts, labels)
    results, summary = evaluate_models(X_train, X_test, y_train, y_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    (output_dir / "classification_reports.json").write_text(
        json.dumps(
            {
                name: result["classification_report"]
                for name, result in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "dataset": Path(data_path).name,
                "total_rows": int(len(frame)),
                "binary_rows": int(len(texts)),
                "excluded_neutral_rows": neutral_count,
                "class_counts": {
                    str(label): int(count)
                    for label, count in labels.value_counts().sort_index().items()
                },
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "tfidf_max_features": MAX_FEATURES,
                "holdout_warning": (
                    "The local dataset is very small; holdout metrics are "
                    "illustrative and not reliable estimates of generalization."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "data": frame,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "summary": summary,
        "neutral_count": neutral_count,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TF-IDF sentiment classifiers on customer reviews."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--text",
        help="Optionally classify one custom review with the best holdout F1 model.",
    )
    args = parser.parse_args()

    run = run_benchmark(args.data, args.output_dir)
    print("Holdout performance (sorted by positive-class F1):")
    print(run["summary"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"\nReviews: {len(run['data'])} total; "
        f"{run['neutral_count']} neutral review(s) excluded from binary training."
    )
    print("Warning: this dataset is too small for reliable generalization estimates.")
    print(f"Metrics saved to: {run['output_dir'].resolve()}")
    if args.text:
        best_name = str(run["summary"].iloc[0]["model"])
        prediction = predict_sentiment(args.text, run["results"][best_name]["estimator"])
        print(f"\n{best_name} prediction: {prediction}")


if __name__ == "__main__":
    main()

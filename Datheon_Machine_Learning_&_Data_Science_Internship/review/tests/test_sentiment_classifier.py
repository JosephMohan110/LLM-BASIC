from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sentiment_classifier import (
    load_dataset,
    predict_sentiment,
    prepare_binary_dataset,
    preprocess_text,
    run_benchmark,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]


class SentimentClassifierTests(unittest.TestCase):
    def test_dataset_schema_and_neutral_exclusion(self) -> None:
        frame = load_dataset(PROJECT_DIR / "data.csv")
        texts, labels, neutral_count = prepare_binary_dataset(frame)

        self.assertEqual(len(frame), 10)
        self.assertEqual(neutral_count, 1)
        self.assertEqual(labels.value_counts().to_dict(), {"positive": 5, "negative": 4})
        self.assertEqual(len(texts), 9)

    def test_preprocessing_removes_markup_and_stopwords_but_keeps_negation(self) -> None:
        cleaned = preprocess_text("<b>The reviews are not terrible!</b>")

        self.assertEqual(cleaned, "review not terrible")

    def test_binary_training_requires_enough_examples_per_class(self) -> None:
        frame = pd.DataFrame(
            {
                "review_id": [1, 2, 3],
                "review_text": ["good", "bad", "neutral"],
                "sentiment": ["positive", "negative", "neutral"],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least two examples"):
            prepare_binary_dataset(frame)

    def test_benchmark_saves_metrics_and_predicts_custom_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run = run_benchmark(PROJECT_DIR / "data.csv", output_dir)

            expected_files = {
                "performance_summary.csv",
                "classification_reports.json",
                "benchmark_metadata.json",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)
            self.assertEqual(
                set(run["results"]),
                {"Multinomial Naive Bayes", "Logistic Regression"},
            )
            for metrics in run["summary"].to_dict(orient="records"):
                self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
                self.assertTrue(0.0 <= metrics["f1_positive"] <= 1.0)

            model = run["results"]["Multinomial Naive Bayes"]["estimator"]
            prediction = predict_sentiment("I love this excellent product", model)
            self.assertIn(prediction["sentiment"], {"positive", "negative"})
            self.assertTrue(0.0 <= prediction["confidence"] <= 1.0)
            metadata = json.loads(
                (output_dir / "benchmark_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["excluded_neutral_rows"], 1)

    def test_prediction_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_benchmark(PROJECT_DIR / "data.csv", Path(temp_dir))
            model = run["results"]["Multinomial Naive Bayes"]["estimator"]

            with self.assertRaisesRegex(ValueError, "non-empty"):
                predict_sentiment("  ", model)


if __name__ == "__main__":
    unittest.main()

# Customer Review Sentiment Classification

An end-to-end text classification project using the local `data.csv`. It cleans and lemmatizes reviews, builds TF-IDF features (`max_features=5000`), and compares Multinomial Naive Bayes with Logistic Regression.

## Project layout

```text
review/
├── data.csv
├── sentiment_classifier.py
├── requirements.txt
├── notebooks/
│   └── sentiment_classification.ipynb
├── outputs/
│   ├── benchmark_metadata.json
│   ├── classification_reports.json
│   └── performance_summary.csv
└── tests/
    └── test_sentiment_classifier.py
```

## Setup on Windows

From PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

The WordNet download is only needed if NLTK does not already have it. The benchmark itself reads the local CSV and makes no network requests.

## Run the notebook

```powershell
jupyter lab
```

Open `notebooks/sentiment_classification.ipynb` and run the cells in order. It loads the local data, reports the class balance, benchmarks both models, and provides `predict_sentiment()` for custom phrases.

## Run the benchmark script

```powershell
python sentiment_classifier.py
python sentiment_classifier.py --text "The product works perfectly and I love it"
```

Optional paths:

```powershell
python sentiment_classifier.py --data data.csv --output-dir outputs
```

## Method and dataset limitations

- The CSV has 10 reviews: 4 positive, 5 negative, and 1 neutral. Since this task is specifically binary, neutral examples are reported but excluded from training and evaluation. The source CSV is not modified.
- Preprocessing removes HTML, lowercases, tokenizes alphabetic words, removes English stopwords while retaining negations (`no`, `not`, `never`, `nor`), and lemmatizes with NLTK WordNet.
- An 75/25 stratified holdout split with seed `42` is used. TF-IDF is fitted inside each model pipeline on training text only. The reported metrics include accuracy and positive-class precision, recall, and F1.
- Logistic Regression is included as a comparison; Multinomial Naive Bayes is the primary requested classifier.
- Ten examples are far too few for reliable accuracy estimates. Holdout metrics can change substantially with one review and should only be treated as an educational smoke-test. Use a much larger, representative labeled dataset and cross-validation before drawing conclusions.

## Tests

```powershell
python -m unittest discover -s tests -v
```

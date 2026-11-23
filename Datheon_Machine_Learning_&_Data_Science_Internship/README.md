# Datheon Machine Learning & Data Science Internship

This repository contains three self-contained machine learning projects completed during the Datheon Machine Learning and Data Science internship. Each project uses a local CSV dataset, includes an executable Python benchmark, a Jupyter notebook, tests, and generated results.

## Projects

| Project | Folder | Task | Models / methods | Dataset |
| --- | --- | --- | --- | --- |
| California Housing Regression | [Estate](Estate/) | Predict median house values | OLS, Ridge, and Lasso | 20,640 housing records |
| Iris Classification | [Iris](Iris/) | Classify iris species | Logistic Regression and Decision Tree | 150 labeled flowers |
| Customer Review Sentiment | [review](review/) | Classify review sentiment | Multinomial Naive Bayes and Logistic Regression | 10 labeled reviews |

## Repository structure

```text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── Estate/
│   ├── data.csv
│   ├── housing_regression.py
│   ├── requirements.txt
│   ├── notebooks/
│   ├── outputs/
│   └── tests/
├── Iris/
│   ├── data.csv
│   ├── iris_classification.py
│   ├── requirements.txt
│   ├── notebooks/
│   ├── outputs/
│   └── tests/
├── review/
│   ├── data.csv
│   ├── sentiment_classifier.py
│   ├── requirements.txt
│   ├── notebooks/
│   ├── outputs/
│   └── tests/
└── README.md
```

## Requirements

- Windows 10 or 11
- Python 3.10 or later
- PowerShell
- JupyterLab

The project dependencies are listed in each project folder's `requirements.txt`.

## Setup

Open PowerShell at the repository root and create a shared virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r Estate\requirements.txt
python -m pip install -r Iris\requirements.txt
python -m pip install -r review\requirements.txt
```

The review classifier uses NLTK's WordNet corpus. Download it once if it is not already available:

```powershell
python -c "import nltk; nltk.download('wordnet')"
```

## Run the projects

### Housing price regression

```powershell
cd Estate
python housing_regression.py
jupyter lab
```

The benchmark compares OLS, Ridge, and Lasso, evaluates them on a held-out test set, performs cross-validated regularization selection, and writes metrics and plots to `Estate\outputs`.

### Iris classification

```powershell
cd ..\Iris
python iris_classification.py
jupyter lab
```

The benchmark uses a stratified holdout, compares Logistic Regression and Decision Tree models, and writes metrics, confusion matrices, ROC curves, and exploratory plots to `Iris\outputs`.

### Review sentiment classification

```powershell
cd ..\review
python sentiment_classifier.py
python sentiment_classifier.py --text "The product works perfectly and I love it"
jupyter lab
```

The script preprocesses review text, performs TF-IDF classification, compares Naive Bayes and Logistic Regression, and writes results to `review\outputs`.

To return to the repository root from the `review` folder:

```powershell
cd ..
```

## Benchmark summaries

The exact checked-in outputs are available in each project's `outputs` folder.

### Housing regression

The models are evaluated on 4,128 holdout rows using RMSE, MAE, and R².

| Model | Test RMSE | Test MAE | Test R² |
| --- | ---: | ---: | ---: |
| OLS | $68,679.61 | $49,133.69 | 0.640 |
| Ridge | $68,687.33 | $49,144.61 | 0.640 |
| Lasso | $69,151.10 | $49,621.15 | 0.635 |

The target is capped at $500,001, and `total_bedrooms` contains 207 missing values. Both issues should be considered when interpreting the regression results.

### Iris classification

Both models achieve 93.3% accuracy on the 30-row holdout set.

| Model | Accuracy | Macro F1 | Macro OvR ROC AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.933 | 0.933 | 0.997 |
| Decision Tree | 0.933 | 0.933 | 0.950 |

### Review sentiment

The source data contains 10 reviews: five positive, four negative, and one neutral. Neutral reviews are excluded from binary training and evaluation.

| Model | Accuracy | Positive precision | Positive recall | Positive F1 |
| --- | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 0.667 | 1.000 | 0.500 | 0.667 |
| Logistic Regression | 0.333 | 0.000 | 0.000 | 0.000 |

This sentiment result is only an educational smoke test because the dataset has just nine binary examples.

## Run all tests

Run each project's test suite from its own directory:

```powershell
cd Estate
python -m unittest discover -s tests -v

cd ..\Iris
python -m unittest discover -s tests -v

cd ..\review
python -m unittest discover -s tests -v
```

## General notes

- Scripts and notebooks read local files and do not require hosted APIs or network requests.
- Random seeds and held-out splits are fixed where applicable to make benchmark runs reproducible.
- The Iris and housing benchmarks use representative but relatively small or historical datasets; no production deployment claims should be inferred.
- The housing data has a capped target, the Iris holdout is only 30 rows, and the review data contains only 10 records. Cross-validation and larger representative datasets should be used for more reliable conclusions.

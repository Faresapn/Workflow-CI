# Workflow-CI

MLflow CI/CD pipeline for Mushroom Classification model training.

## Overview

This repository implements an automated CI/CD workflow using **MLflow Project** and **GitHub Actions** to retrain a Random Forest classifier on the Mushroom dataset whenever changes are pushed to the `main` branch.

## Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── workflow-ci.yml          # GitHub Actions CI workflow
├── MLProject/
│   ├── modelling.py                  # Training script (Random Forest)
│   ├── conda.yaml                    # Environment dependencies
│   ├── MLProject                     # MLflow project config
│   └── mushrooms_preprocessed/
│       └── mushrooms_preprocessed.csv # Preprocessed dataset
├── .gitignore
└── README.md
```

## How It Works

1. **Trigger:** On every `push` or `pull_request` to `main` branch
2. **Setup:** Python 3.9 environment with MLflow and scikit-learn
3. **Training:** Runs `mlflow run MLProject/` to train the model
4. **Artifacts:** Uploads trained model and MLflow runs as GitHub Actions artifacts

## Local Usage

```bash
# Install dependencies
pip install mlflow scikit-learn pandas numpy

# Run training locally
cd MLProject
mlflow run . --no-conda
```

## Model Details

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Dataset:** Mushroom Classification (preprocessed)
- **Split:** 80/20 train/test (stratified)
- **Metrics:** Accuracy, Precision, Recall, F1-Score

## GitHub Actions

The workflow runs automatically and:
- Trains the model using MLflow
- Logs metrics and parameters via `mlflow.autolog()`
- Uploads model artifacts for download

## Author

**Faresa Prasetyo Nugroho** — @Faresapn

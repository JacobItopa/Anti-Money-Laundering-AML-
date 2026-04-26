# AML Fraud Detection — End-to-End ML Pipeline

[![CI — Tests & Lint](https://github.com/your-username/aml-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/aml-fraud-detection/actions/workflows/ci.yml)
[![CD — Deploy to Cloud Run](https://github.com/your-username/aml-fraud-detection/actions/workflows/cd.yml/badge.svg)](https://github.com/your-username/aml-fraud-detection/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade Anti-Money Laundering (AML) fraud detection system built across the full ML lifecycle — from raw data exploration to a containerised REST API — incorporating Logistic Regression, XGBoost, and a Graph Neural Network (GraphSAGE) for benchmarking.

---

## Table of Contents

- [Project Overview](#project-overview)
- [The Dataset & The Challenge](#the-dataset--the-challenge)
- [Model Results & Key Findings](#model-results--key-findings)
- [Project Structure](#project-structure)
- [Step-by-Step Documentation](#step-by-step-documentation)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [API Usage](#api-usage)
- [Testing](#testing)
- [CI/CD/CT Pipeline (GitHub Actions)](#cicdct-pipeline-github-actions)
- [Deployment to GCP Cloud Run](#deployment-to-gcp-cloud-run)

---

## Project Overview

Financial crime costs the global economy an estimated **$2 trillion per year**. This project implements a complete machine learning pipeline to automatically flag suspicious transactions that may constitute money laundering. The system is trained on the [IBM AML synthetic dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and outputs a real-time fraud probability score for any incoming financial transaction via a REST API.

The project follows the **Cookiecutter Data Science** structure, with a strict separation between exploration notebooks and production-grade `src/` modules.

---

## The Dataset & The Challenge

| Property | Value |
|---|---|
| **Dataset** | IBM Synthetic AML Transactions (`HI-Small_Trans.csv`) |
| **Total Transactions** | ~5.08 million |
| **Illicit Transactions** | ~5,184 (≈ 0.1%) |
| **Class Imbalance Ratio** | **1 : 980** (Normal : Illicit) |
| **Primary Metric** | **PR-AUC** (Precision-Recall AUC) |

The extreme 1:980 class imbalance is the central engineering challenge of this project. Standard accuracy is meaningless (a model predicting "normal" for everything scores 99.9%). We use **Precision-Recall AUC** as our North Star metric, as it directly captures the trade-off between catching launderers (Recall) and the rate of false alarms (Precision).

---

## Model Results & Key Findings

We benchmarked three fundamentally different model families across the full dataset and a 500k-transaction subgraph.

### Full Dataset (1M-row holdout test set)

| Model | PR-AUC | Recall (Illicit) | Notes |
|---|---|---|---|
| Random Guessing | ~0.001 | — | Theoretical baseline for 1:980 ratio |
| **Logistic Regression** | 0.011 | — | `class_weight='balanced'` |
| **XGBoost** | **0.068** | **90%** | `scale_pos_weight=980`, `tree_method='hist'` |

**Finding 1:** XGBoost achieves a **68× improvement over random guessing** and correctly identifies **90% of all real money laundering transactions** on the unseen test set. The 1% Precision is a known trade-off: the model acts as a highly sensitive dragnet, surfacing suspicious cases for a human compliance analyst to investigate.

**Finding 2:** Hyperparameter tuning via `RandomizedSearchCV` on a 500k-row sample found optimal parameters (`max_depth=4`, `learning_rate=0.1`), but did not improve PR-AUC over the base model. This demonstrates a classic ML principle: **more data beats better parameters**.

### Subgraph Comparison (500k transactions)

| Model | PR-AUC | Architecture |
|---|---|---|
| XGBoost | 0.1904 | Tabular (row-level) |
| **GraphSAGE GNN** | **0.2914** | Graph (topology-aware) |

**Finding 3 (Breakthrough):** Our Graph Neural Network, which "sees" the flow of money between accounts as a connected graph rather than isolated rows, achieved a **~50% relative improvement in PR-AUC over XGBoost** on the same transactions. This definitively proves that network topology (cyclic layering patterns, gather-scatter hubs) is a highly predictive signal for money laundering. In a full production environment with GPU cluster resources, the GNN is the clear production model.

---

## Project Structure

```
AML/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Runs test suite on every push/PR
│       ├── cd.yml              # Deploys to GCP Cloud Run on merge to main
│       └── ct.yml              # Monthly drift check + conditional retraining
│
├── data/
│   ├── raw/                    # Original, immutable IBM AML dataset
│   ├── interim/                # Serialised train/test splits (parquet)
│   └── processed/              # ML-ready feature dataset (parquet)
│
├── models/                     # Serialised model artefacts (.joblib)
│
├── notebooks/                  # Exploratory Jupyter notebooks
│   ├── 01-eda.ipynb
│   ├── 02-eda-visualizations.ipynb
│   ├── 03-model-training.ipynb
│   ├── 04-gnn-comparison.ipynb
│   └── 05-evaluation-tuning.ipynb
│
├── reports/
│   └── figures/                # Auto-generated PR curves, confusion matrices
│
├── src/
│   ├── app/
│   │   └── main.py             # FastAPI REST API with latency middleware
│   ├── features/
│   │   └── build_features.py   # Full preprocessing & feature engineering pipeline
│   └── models/
│       ├── train_model.py      # Logistic Regression & XGBoost training
│       ├── train_gnn.py        # GraphSAGE GNN training (PyTorch Geometric)
│       ├── evaluate_model.py   # PR-AUC, confusion matrix, PR-curve plotting
│       ├── tune_model.py       # RandomizedSearchCV hyperparameter tuning
│       └── drift_detection.py  # KS-test data drift detection
│
├── tests/
│   ├── test_build_features.py  # 21 unit tests for feature pipeline
│   ├── test_drift_detection.py # 3 unit tests for drift detection
│   └── test_api_integration.py # 16 integration tests for the REST API
│
├── Dockerfile                  # Production container definition
├── .dockerignore
├── deploy_gcp.sh               # One-shot GCP Cloud Run deploy script
├── generate_artifacts.py       # Extracts preprocessing artefacts for the API
├── retrain_pipeline.py         # CT orchestration — re-runs full pipeline
├── run_preprocessing.py        # Runs build_features pipeline
├── run_training.py             # Trains and serialises models
├── run_evaluation.py           # Evaluates + plots all models
├── run_gnn.py                  # Trains GNN and compares to XGBoost
└── requirements.txt
```

---

## Step-by-Step Documentation

Each step of the ML lifecycle has a dedicated markdown report documenting the methodology, design decisions, and findings:

| File | Description |
|---|---|
| [`step_1_problem_definition.md`](step_1_problem_definition.md) | Defines the AML fraud detection problem, business objective, class imbalance analysis, and why PR-AUC was chosen as the primary evaluation metric. |
| [`step_2_data_collection.md`](step_2_data_collection.md) | Describes the IBM AML dataset, its provenance, schema, and initial profiling. |
| [`step_3_eda.md`](step_3_eda.md) | Documents the Exploratory Data Analysis findings, including laundering typologies, temporal patterns, and distribution analysis. |
| [`step_4_preprocessing.md`](step_4_preprocessing.md) | Details the full feature engineering pipeline: temporal extraction, log-scaling for skewed amounts, frequency encoding for high-cardinality banks/currencies, and one-hot encoding. |
| [`step_5_training.md`](step_5_training.md) | Documents the training methodology and **performance results** for all three models: Logistic Regression, XGBoost, and the GraphSAGE GNN, including the 50% relative GNN improvement. |
| [`step_6_evaluation.md`](step_6_evaluation.md) | Covers the formal evaluation on the 1M-row test set, hyperparameter tuning with RandomizedSearchCV, the generated PR-Curve comparisons, and confusion matrices. |
| [`step_7_deployment.md`](step_7_deployment.md) | Explains the MLOps architecture: how preprocessing state is preserved for single-row inference via serialised artefacts, the FastAPI API design, Docker containerisation strategy, and the GCP Cloud Run deployment playbook. |
| [`step_8_monitoring.md`](step_8_monitoring.md) | Defines the full monitoring strategy: system latency tracking (middleware), KS-test data drift detection, the distinction between Data Drift and Concept Drift in financial fraud, and the CI/CD/CT retraining strategy. |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)
- `gcloud` CLI (for GCP deployment only)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/aml-fraud-detection.git
cd aml-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the raw dataset
# Download HI-Small_Trans.csv from Kaggle and place it in:
# data/raw/HI-Small_Trans.csv
```

---

## Running the Pipeline

Run the scripts in order to reproduce all results from scratch:

```bash
# Step 1: Feature engineering → generates data/processed/processed_transactions.parquet
python run_preprocessing.py

# Step 2: Train Logistic Regression + XGBoost → saves to models/
python run_training.py

# Step 3: Extract preprocessing artefacts for the API
python generate_artifacts.py

# Step 4: Evaluate models + generate figures
python run_evaluation.py

# Step 5: Train GNN and compare against XGBoost (runs on 500k subgraph)
python run_gnn.py
```

---

## API Usage

### Start the server locally

```bash
python -m uvicorn src.app.main:app --port 8080
```

### Send a prediction request

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Timestamp": "2022/09/01 00:20",
    "From_Bank": "10",
    "Account": "8000EBD30",
    "To_Bank": "10",
    "Account_1": "8000EBD30",
    "Amount_Received": 10.50,
    "Receiving_Currency": "US Dollar",
    "Amount_Paid": 10.50,
    "Payment_Currency": "US Dollar",
    "Payment_Format": "Reinvestment"
  }'
```

### Example response

```json
{
  "fraud_probability": 0.000020,
  "is_laundering": false,
  "flagged_for_review": false
}
```

The API also returns an `X-Process-Time` header on every response for latency tracking.

**Interactive API docs:** `http://127.0.0.1:8080/docs` (FastAPI auto-generated Swagger UI)

---

## Testing

The project includes **39 tests** across unit and integration test suites:

```bash
python -m pytest tests/ -v
```

| Test File | Type | Tests | What's Covered |
|---|---|---|---|
| `test_build_features.py` | Unit | 21 | Every feature engineering function in isolation |
| `test_drift_detection.py` | Unit | 3 | KS-test drift detection with synthetic data |
| `test_api_integration.py` | Integration | 16 | Full request lifecycle, validation, edge cases, latency headers |

---

## CI/CD/CT Pipeline (GitHub Actions)

Three automated workflows are defined in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|---|---|---|
| **`ci.yml`** | Push / Pull Request to `main` or `develop` | Installs deps, generates artefacts, runs all 39 tests |
| **`cd.yml`** | Merge to `main` | Builds Docker image, pushes to GCR, deploys to Cloud Run |
| **`ct.yml`** | Monthly cron + manual dispatch | Runs drift detection; retrains full pipeline if drift is detected |

---

## Deployment to GCP Cloud Run

Once your GCP subscription is active:

### Option A — Automated (recommended)

1. In your GitHub repository, go to **Settings → Secrets and variables → Actions**
2. Add two secrets:
   - `GCP_PROJECT_ID` — your Google Cloud Project ID
   - `GCP_SA_KEY` — the JSON key of a service account with **Cloud Run Admin** and **Storage Admin** roles
3. Merge any change to `main` — the CD workflow deploys automatically.

### Option B — Manual script

```bash
# Edit deploy_gcp.sh and set your PROJECT_ID, then uncomment the commands
bash deploy_gcp.sh
```

---

## License

This project is licensed under the MIT License.

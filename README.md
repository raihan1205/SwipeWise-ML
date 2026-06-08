# SwipeWise — Predicting the Fate of a Match

Group project for **WIA1006 / WID3006 Machine Learning** (Sem 2, 2025/2026).
A full ML pipeline predicting `match_outcome` (10 classes) from a synthetic dating-app behaviour dataset.

> **Group members:** _add names here_

## Problem
Multi-class classification of dating-app `match_outcome` (e.g. Mutual Match, Ghosted, Catfished).
Dataset: 50,000 records × 25 features (Kaggle "dating-app-behaviour" extended).

## Pipeline / Notebooks
Run in order — notebook 1 produces the train/test splits the others load.

| # | Notebook | Stage |
|---|----------|-------|
| 1 | `notebooks/1_SwipeWise_EDA_Preprocessing.ipynb` | EDA, preprocessing, feature engineering & selection |
| 2 | `notebooks/2_SwipeWise_Baselines_Models.ipynb` | Baselines + Decision Tree, Random Forest, KNN |
| 3 | `notebooks/3_SwipeWise_Advanced_Tuning.ipynb` | Tuned Random Forest & Gradient Boosting |
| 4 | `notebooks/4_SwipeWise_Evaluation.ipynb` | Evaluation, comparison & auto-sklearn benchmark |

## Data
- `data/dating_app_behavior_dataset_extended1.csv` — raw input (committed).
- `X_train.csv`, `X_test.csv`, `y_*.csv`, `*.joblib` — **generated** by notebook 1, so they are git-ignored. Run notebook 1 to recreate them.

## Key finding
All models converge to ~10% accuracy — exactly chance for a 10-class balanced target.
A feature-vs-target independence test (Cramér's V ≈ 0.01 for every feature) confirms the
synthetic labels carry **no learnable signal**. The pipeline is correct; the dataset has no signal.

## How to run
1. Open the notebooks in Google Colab or Jupyter.
2. Run notebook 1 fully to generate the data splits.
3. Run notebooks 2 → 4 (upload the generated CSVs if using Colab).
4. `auto-sklearn` (notebook 4) requires Linux / Colab.

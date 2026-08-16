"""
train_model.py — fits a win-probability model on NFL play-by-play data
and evaluates it properly, not just on accuracy.

Split strategy: train on 2021-2024, test on the full 2025 season held
out entirely. Splitting by season (not by random play) avoids leaking
plays from the same game across train/test — plays within a game are
highly correlated, so a random row-level split would silently inflate
every metric below.

Fits both a logistic regression baseline and an XGBoost model, and
compares them on calibration (Brier score + reliability curve), not
just accuracy — a model can be 70% accurate and still be badly
miscalibrated, which is the part that actually matters for a "sports
modeling" role: nobody cares if you got the winner right, they care if
your 73% meant 73%.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from features import FEATURE_COLUMNS, get_X_y

DATA_PATH = Path(__file__).parent / "data" / "pbp_model_data.parquet"
MODEL_PATH = Path(__file__).parent / "data" / "wp_model.xgb"
METRICS_PATH = Path(__file__).parent / "data" / "metrics.json"

TEST_SEASON = 2025


def evaluate(y_true, p_pred, label: str) -> dict:
    metrics = {
        "brier_score": float(brier_score_loss(y_true, p_pred)),
        "log_loss": float(log_loss(y_true, p_pred)),
        "auc": float(roc_auc_score(y_true, p_pred)),
    }
    print(f"\n{label}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    return metrics


def main():
    raw = pd.read_parquet(DATA_PATH)
    X, y, prepped = get_X_y(raw)

    train_mask = prepped["season"] < TEST_SEASON
    test_mask = prepped["season"] == TEST_SEASON

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Train: {len(X_train)} plays ({sorted(prepped.loc[train_mask,'season'].unique())})")
    print(f"Test : {len(X_test)} plays (season {TEST_SEASON}, fully held out)")

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)
    p_logit = logit.predict_proba(X_test)[:, 1]
    metrics_logit = evaluate(y_test, p_logit, "Logistic Regression (baseline)")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    p_xgb = model.predict_proba(X_test)[:, 1]
    metrics_xgb = evaluate(y_test, p_xgb, "XGBoost")

    # calibration curve for both models — not used for training, only
    # saved to metrics.json so the dashboard can plot a reliability curve
    frac_pos_logit, mean_pred_logit = calibration_curve(y_test, p_logit, n_bins=10, strategy="quantile")
    frac_pos_xgb, mean_pred_xgb = calibration_curve(y_test, p_xgb, n_bins=10, strategy="quantile")

    results = {
        "test_season": TEST_SEASON,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "logistic_regression": metrics_logit,
        "xgboost": metrics_xgb,
        "calibration": {
            "logistic_regression": {"predicted": mean_pred_logit.tolist(), "actual": frac_pos_logit.tolist()},
            "xgboost": {"predicted": mean_pred_xgb.tolist(), "actual": frac_pos_xgb.tolist()},
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"\nXGBoost beat logistic regression on Brier score: "
          f"{metrics_xgb['brier_score']:.4f} vs {metrics_logit['brier_score']:.4f} "
          f"({'better' if metrics_xgb['brier_score'] < metrics_logit['brier_score'] else 'worse'} = lower is better)")


if __name__ == "__main__":
    main()

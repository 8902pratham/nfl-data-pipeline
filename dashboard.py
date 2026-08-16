"""
dashboard.py — explore the win probability model.

Two views:
  1. Pick any real 2025 game, see the win probability curve play out
     drive by drive.
  2. The calibration comparison (logistic regression vs XGBoost) that
     validates the model actually means what it says — a reliability
     curve, not just a leaderboard metric.

Run with: streamlit run dashboard.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from features import FEATURE_COLUMNS, get_X_y

DATA_PATH = Path(__file__).parent / "data" / "pbp_model_data.parquet"
MODEL_PATH = Path(__file__).parent / "data" / "wp_model.xgb"
METRICS_PATH = Path(__file__).parent / "data" / "metrics.json"

st.set_page_config(page_title="NFL Win Probability Model", page_icon="🏈", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    raw = pd.read_parquet(DATA_PATH)
    X, y, prepped = get_X_y(raw)
    model = load_model()
    prepped = prepped.assign(pred=model.predict_proba(X)[:, 1])
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return prepped, metrics


def game_view(prepped: pd.DataFrame):
    season_2025 = prepped[prepped.season == 2025]
    games = sorted(season_2025.game_id.unique(), reverse=True)
    game_id = st.selectbox("Pick a 2025 game", games)

    g = season_2025[season_2025.game_id == game_id].sort_values("game_seconds_remaining", ascending=False).copy()

    away, home = game_id.split("_")[2], game_id.split("_")[3]
    final_margin = g.result.iloc[0]  # home - away
    winner = home if final_margin > 0 else away
    st.write(f"**{away} @ {home}** — final margin {abs(final_margin):.0f} for **{winner}**")

    # convert posteam-perspective win prob to home-team perspective for a
    # single consistent line on the chart
    g["home_win_prob"] = g.apply(
        lambda r: r["pred"] if r["posteam"] == home else 1 - r["pred"], axis=1
    )
    g["play_number"] = range(len(g))

    st.line_chart(g.set_index("play_number")["home_win_prob"], height=350)
    st.caption(f"Win probability for {home} (home team) across the game. 0.50 = coin flip.")

    with st.expander("Raw plays behind this chart"):
        st.dataframe(
            g[["qtr", "down", "ydstogo", "posteam", "score_differential", "home_win_prob"]].reset_index(drop=True),
            use_container_width=True,
        )


def calibration_view(metrics: dict):
    st.write(
        f"Tested on the full **{metrics['test_season']} season held out entirely** "
        f"({metrics['n_test']:,} plays) — trained only on {metrics['n_train']:,} plays from prior seasons. "
        "No game in the test set was ever seen in training."
    )

    col1, col2 = st.columns(2)
    for col, key, label in [(col1, "logistic_regression", "Logistic Regression"), (col2, "xgboost", "XGBoost")]:
        m = metrics[key]
        with col:
            st.metric(f"{label} — Brier score", f"{m['brier_score']:.4f}", help="Lower is better. 0 = perfect.")
            st.metric(f"{label} — AUC", f"{m['auc']:.4f}")

    st.subheader("Calibration: predicted probability vs. actual outcome rate")
    st.caption(
        "If the model is well-calibrated, plays where it predicts ~70% win probability "
        "should actually be won about 70% of the time. Points on the diagonal = perfectly calibrated."
    )

    cal_df = pd.DataFrame({
        "Predicted probability": metrics["calibration"]["xgboost"]["predicted"],
        "XGBoost — actual rate": metrics["calibration"]["xgboost"]["actual"],
        "Logistic Regression — actual rate": metrics["calibration"]["logistic_regression"]["actual"],
        "Perfect calibration": metrics["calibration"]["xgboost"]["predicted"],
    })
    st.line_chart(cal_df.set_index("Predicted probability"), height=400)


def main():
    st.title("🏈 NFL Win Probability Model")
    st.caption(
        "XGBoost model trained on 2021-2024 play-by-play, predicting live win probability "
        "from game state (score, time, down/distance, field position, timeouts)."
    )

    prepped, metrics = load_data()

    tab1, tab2 = st.tabs(["Game Explorer", "Model Validation"])
    with tab1:
        game_view(prepped)
    with tab2:
        calibration_view(metrics)


if __name__ == "__main__":
    main()

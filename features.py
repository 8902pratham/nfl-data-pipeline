"""
features.py — shared data prep for the win probability model.

Both train_model.py and dashboard.py import from here so the exact same
filtering and feature logic is used at train time and inference time.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "score_differential",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter raw play-by-play down to normal snaps with a clean win/loss
    label, and attach the label column `posteam_won`.

    Excludes: overtime (different clock rules), plays without a down
    (kickoffs, extra points, penalties with no play), and games that
    ended in a tie (rare, and undefined for a binary win label).
    """
    df = df.copy()

    # normal snaps only — this also drops the ~5-6% of rows with nulls
    # in score_differential/timeouts/posteam that come from kickoffs etc.
    df = df[df["game_half"] != "Overtime"]
    df = df.dropna(subset=FEATURE_COLUMNS + ["posteam", "posteam_type", "result"])

    # drop tied games — result is the final home-minus-away margin
    df = df[df["result"] != 0]

    home_won = df["result"] > 0
    posteam_is_home = df["posteam_type"] == "home"
    df["posteam_won"] = (home_won & posteam_is_home) | (~home_won & ~posteam_is_home)
    df["posteam_won"] = df["posteam_won"].astype(int)

    return df


def get_X_y(df: pd.DataFrame):
    prepped = prepare(df)
    return prepped[FEATURE_COLUMNS], prepped["posteam_won"], prepped

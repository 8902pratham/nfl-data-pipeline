"""
fetch_pbp.py — pulls play-by-play data for a set of seasons from nflverse
and saves a lightweight, modeling-ready parquet file.

Source: nflverse-data GitHub releases (same canonical source nfl_data_py
wraps for play-by-play). Each season is ~370 columns / ~20MB; we only keep
what the win-probability model needs.
"""

from pathlib import Path

import pandas as pd

SEASONS = [2021, 2022, 2023, 2024, 2025]  # 5 most recent complete seasons
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
OUT_PATH = Path(__file__).parent / "data" / "pbp_model_data.parquet"

KEEP_COLUMNS = [
    "game_id", "season", "week", "season_type", "game_half",
    "posteam", "defteam", "posteam_type",
    "score_differential", "game_seconds_remaining", "half_seconds_remaining",
    "qtr", "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "play_type", "result",
]


def fetch_season(season: int) -> pd.DataFrame:
    url = PBP_URL.format(season=season)
    df = pd.read_parquet(url, columns=KEEP_COLUMNS)
    return df


def main():
    frames = []
    for season in SEASONS:
        print(f"Fetching {season} play-by-play ...")
        df = fetch_season(season)
        print(f"  {len(df)} plays")
        frames.append(df)

    all_pbp = pd.concat(frames, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_pbp.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(all_pbp)} total plays across {len(SEASONS)} seasons to {OUT_PATH}")
    print(f"File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

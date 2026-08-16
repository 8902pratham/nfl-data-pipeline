# NFL Win Probability Model

Predicts live win probability from in-game state (score, time remaining,
down/distance, field position, timeouts) — the same kind of model that
sits behind betting lines and TV win-probability graphics. Built as a
portfolio piece for sports-modeling roles (Swish Analytics, DraftKings,
FanDuel-style teams) and team analytics departments alike.

**Live dashboard:** _add your Streamlit Cloud link here after deploying_

## How it works — data flow

Four scripts, each doing one job, connected by two files on disk
(`pbp_model_data.parquet` and the trained model/metrics). Nothing here
is a class — it's a small functional pipeline, so the diagram below is
really a map of *functions*, not objects.

```
nflverse-data (GitHub releases)
        │   play_by_play_{season}.parquet  ×  2021-2025
        ▼
fetch_pbp.py
  fetch_season(season) → pd.DataFrame   (downloads 1 season, keeps 18 cols)
  main()                                (loops seasons, concatenates, writes parquet)
        ▼
data/pbp_model_data.parquet        ← raw play-by-play, unfiltered, ~206k rows
        ▼
features.py                        (imported by BOTH scripts below —
  prepare(df)                       this is what keeps train-time and
    → drops overtime, drops rows    inference-time feature logic from
      missing a feature/label,      silently drifting apart)
      drops tied games, adds
      posteam_won label
  get_X_y(df)
    → calls prepare(), returns
      (X, y, prepped)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
train_model.py                                                dashboard.py
  main()                                                        (re-runs get_X_y on the
    • season < 2025 → train                                      FULL parquet at load
    • season == 2025 → test (held out entirely)                  time, then scores every
    • fits LogisticRegression (baseline)                         play with the pickled
    • fits XGBoost (main model)                                  model — this is the
  evaluate(y_true, p_pred, label)                                "inference" half of the
    → Brier score, log loss, AUC for one model                   pipeline)
    • also computes a calibration_curve for each model
        │
        ├──▶ data/wp_model.xgb      (pickled XGBoost model)
        └──▶ data/metrics.json      (both models' scores + calibration curves)
                    │
                    ▼
              dashboard.py  (streamlit run dashboard.py)
  load_model()      → unpickles wp_model.xgb                 (cached)
  load_data()       → get_X_y() on full parquet + model.predict_proba()  (cached)
  game_view(prepped)
    → lets you pick one 2025 game, plots home-team win probability
      play by play, shows the raw plays behind the chart
  calibration_view(metrics)
    → shows Brier/AUC for both models + the reliability curve
      (predicted 70% should mean *won 70% of the time*)
  main()            → wires the two views into two Streamlit tabs
                    ▼
              Streamlit UI (localhost:8501)
```

## Code walkthrough

No custom classes are defined anywhere in this project — `LogisticRegression`
and `xgb.XGBClassifier` are the only "classes" involved, and they're
imported, not written. Everything else below is a plain function.

### `fetch_pbp.py` — data acquisition
| Function | What it does |
|---|---|
| `fetch_season(season: int) -> pd.DataFrame` | Downloads one season's play-by-play parquet directly from the nflverse-data GitHub release, reading only the 18 columns in `KEEP_COLUMNS` (out of ~370 available) to keep the file small. |
| `main()` | Loops over `SEASONS` (2021-2025), calls `fetch_season()` for each, concatenates all five into one DataFrame, and writes it to `data/pbp_model_data.parquet`. Prints row counts and final file size as it goes. |

### `features.py` — shared feature engineering (imported by both training and inference)
| Name | What it does |
|---|---|
| `FEATURE_COLUMNS` | The 7 columns the model actually trains/predicts on: `score_differential`, `game_seconds_remaining`, `down`, `ydstogo`, `yardline_100`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`. |
| `prepare(df) -> pd.DataFrame` | Cleans raw play-by-play into modeling-ready rows: drops overtime plays (different clock rules), drops rows with nulls in any feature column (kickoffs/extra points/no-plays don't have a down), drops tied games (undefined for a binary label), and adds the `posteam_won` column by comparing the final `result` margin to which side (home/away) had possession. |
| `get_X_y(df) -> (X, y, prepped)` | Calls `prepare()`, then splits the result into `X` (the 7 feature columns), `y` (the `posteam_won` label), and `prepped` (the full cleaned DataFrame, kept around because both callers need extra columns like `season` or `game_id` that aren't features). |

### `train_model.py` — training & evaluation
| Function | What it does |
|---|---|
| `evaluate(y_true, p_pred, label) -> dict` | Scores one model's predictions with three metrics — Brier score, log loss, AUC — prints them, and returns them as a dict. Used once per model. |
| `main()` | Loads the parquet, calls `get_X_y()`, splits by season (train: 2021-2024, test: 2025 — held out entirely, see rationale in the module docstring), fits a `LogisticRegression` baseline and an `xgb.XGBClassifier`, scores both with `evaluate()`, computes a calibration curve for each, then pickles the XGBoost model to `wp_model.xgb` and writes everything (scores + calibration curves) to `metrics.json`. |

### `dashboard.py` — Streamlit app
| Function | What it does |
|---|---|
| `load_model()` | Unpickles `wp_model.xgb`. Decorated `@st.cache_resource` so it only runs once per session. |
| `load_data()` | Reads the parquet, runs it through `get_X_y()` (the exact same function `train_model.py` used), loads the model, and adds a `pred` column with each play's win probability. Decorated `@st.cache_data`. |
| `game_view(prepped)` | Renders the "Game Explorer" tab: a dropdown of every 2025 game, a line chart of the **home team's** win probability play-by-play (the model itself predicts from the possessing team's perspective, so this flips it to a single consistent line), and an expandable table of the raw plays behind the chart. |
| `calibration_view(metrics)` | Renders the "Model Validation" tab: Brier score and AUC for both models side by side, plus the calibration/reliability curve loaded straight from `metrics.json` — no recomputation at dashboard time. |
| `main()` | Sets up the page title, loads data via `load_data()`, and puts `game_view()` and `calibration_view()` into two tabs. |

## Results

Trained on 165,285 plays from the 2021-2024 seasons. Tested on 40,371
plays from the **2025 season, held out entirely** — no game in the test
set was seen during training, and the split is by season rather than by
play, so no game has some plays in train and others in test (a play-level
random split would leak information between plays in the same game and
quietly inflate every metric below).

| Model | Brier score ↓ | Log loss ↓ | AUC ↑ |
|---|---|---|---|
| Logistic Regression (baseline) | 0.1742 | 0.5161 | 0.8169 |
| **XGBoost** | **0.1725** | **0.5111** | **0.8204** |

XGBoost wins on all three, which is expected — it captures interactions
like "a 3-point lead means something different at 2 minutes left vs. 2
quarters left" without needing them hand-engineered as features.

Brier score (not just accuracy) is the headline metric on purpose: a
model can correctly pick the winner 70% of the time and still be
miscalibrated garbage. The dashboard's "Model Validation" tab has the
actual reliability curve — predicted 70% should mean *won 70% of the
time*, not just "favored."

## Data & features

Source: [nflverse-data](https://github.com/nflverse/nflverse-data) play-by-play
(the same canonical source `nfl_data_py` wraps). Filtered to normal snaps
(drops kickoffs, extra points, and no-plays, which don't have a
meaningful down/distance) and excludes overtime and tied games.

Features, all pre-game-state, no leakage:
`score_differential`, `game_seconds_remaining`, `down`, `ydstogo`,
`yardline_100`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`.

## Running it

```bash
pip install -r requirements.txt
python fetch_pbp.py     # downloads 5 seasons of play-by-play (~2 min)
python train_model.py   # trains + evaluates both models (~30 sec)
streamlit run dashboard.py
```

`features.py` is imported by both the training script and the dashboard,
so train-time and inference-time feature logic can't silently drift apart
— a common source of bugs in exactly this kind of project.

## Deploying

`data/pbp_model_data.parquet`, `data/wp_model.xgb`, and `data/metrics.json`
are committed to the repo (small — ~2.4MB total), so the dashboard works
straight off a fresh clone with no setup step. Push to GitHub, connect the
repo on [Streamlit Community Cloud](https://streamlit.io/cloud) free tier,
point it at `dashboard.py`, done. Unlike the pipeline project this one
doesn't need daily automation — the model doesn't need retraining on a
schedule, just re-run `fetch_pbp.py` + `train_model.py` and commit the
updated files whenever you want to refresh it (e.g., once 2026 has enough
games to be worth adding).

## Known limitations (worth naming, not hiding)

- **Late-game tail probabilities run slightly optimistic** — e.g., down
  a full touchdown with under 10 seconds left, the model gives more like
  15-18% than the near-0% a human would. This is a known hard spot for
  WP models generally (rare, extreme game states are underrepresented in
  training data) and a good thing to raise proactively in an interview
  rather than let someone else find it.
- No pre-season team-strength prior — two 0-0 teams get ~50/50 regardless
  of whether it's the Chiefs or a rebuilding team. Adding a team-strength
  feature (prior-season point differential, or a simple Elo rating) is
  the highest-value next improvement if you have time.
- Overtime is excluded entirely rather than modeled (different clock and
  win conditions) — noted, not silently dropped.

## Extending this

- Add a team-strength prior (see above) — probably the single biggest
  accuracy improvement available.
- Retrain including 2025 in the training set once 2026 data starts
  coming in, and hold out 2026 instead — keeps the eval honest as new
  data arrives.
- Wire this dashboard up to the data pipeline project: same nflverse
  ecosystem, same Streamlit pattern, could live in the same repo as a
  second page.

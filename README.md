# NBA2627PlayerPredictions

NBA player analytics and 2026-27 season projections built with Python, SQL and Power BI.

The project uses historical player seasons to answer two types of questions:

1. Who performed best in different areas during 2025-26?
2. What could returning NBA players average in 2026-27?

The prediction targets are:

- points per game
- rebounds per game
- assists per game
- 3-pointers made per game
- steals per game
- blocks per game

The analytics side also creates:

- scoring score
- playmaking score
- offense score
- defense score
- all-round score
- 3-point shooting score
- mid-range shooting ranking

## Project flow

```text
NBA Stats
   |
   v
Historical player seasons
   |
   v
Clean player-season table
   |
   +----------------------+
   |                      |
   v                      v
2025-26 analytics     ML feature table
                          |
                          v
                   historical backtest
                          |
                          v
                  2026-27 projections
   |                      |
   +----------+-----------+
              |
              v
         StreamLit
```

## Data

The collection script uses the `nba_api` Python package to access NBA Stats endpoints.

Ten regular seasons are collected:

```text
2016-17 through 2025-26
```

Base statistics include PTS, REB, AST, STL, BLK, shooting and minutes.

Advanced statistics include TS%, usage, offensive rating, defensive rating and rebounding/assist percentages.

The latest season also includes NBA shot-location data for mid-range analysis.

## Why this is player-based

Team rosters change too often for team identity to be a stable prediction feature.

The model therefore uses the player's own production, role, efficiency, age and recent changes. Team is still kept as display information.

## Models

For each target, the project compares:

- Linear Regression
- Random Forest

The 2025-26 season is used as the final historical backtest.

The model with the lower MAE is selected for each statistic, retrained on all available historical pairs and used for 2026-27 projections.

## Run

Mac:

```bash
python3 --version
```

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/run_all.py
```

After the first data download:

```bash
python3 src/run_all.py --skip-download
```

## Main outputs

```text
outputs/predictions_2026_27.csv
outputs/model_metrics.csv
outputs/player_scores_2025_26.csv
outputs/leaderboards_2025_26.csv
outputs/leaderboards_2026_27.csv
outputs/midrange_leaders_2025_26.csv
database/nba_predictions.db
```

## Important limitation

These are statistical projections, not guarantees.

The model does not know future injuries, retirements, unexpected trades or rotation decisions. Rookies without an NBA season are not included in the 2026-27 prediction table.

See `docs/HOW_IT_WORKS.md` for the full explanation.

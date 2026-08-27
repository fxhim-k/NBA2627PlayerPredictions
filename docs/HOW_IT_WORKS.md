# How the project works

## 1. Data collection

`collect_data.py` downloads ten regular seasons of player data from NBA Stats through `nba_api`.

For each season it downloads:

- base/per-game stats
- advanced stats

It also downloads 2025-26 shot-location data for the current mid-range analysis.

The important identifier is `PLAYER_ID`. A player's team can change, but the player ID remains the same.

## 2. Cleaning

`prepare_data.py` keeps only the columns used by the project and merges base and advanced data by `PLAYER_ID`.

The result is:

`data/processed/player_history.csv`

Each row represents one player in one season.

## 3. Feature engineering

`build_features.py` creates a machine-learning row from a player's current season.

Examples of model inputs:

- age
- games played
- minutes
- field-goal attempts
- 3-point attempts and percentage
- free-throw percentage
- current PTS / REB / AST / STL / BLK
- true-shooting percentage
- usage percentage
- offensive and defensive ratings
- change in minutes and production from the previous season

The target is the player's statistic in the following season.

Example:

2024-25 player data -> target: 2025-26 PPG

Only consecutive seasons are used.

## 4. Why team is not a model feature

Trades happen constantly.

The model therefore does not try to learn that a specific team automatically causes a player to score more or less.

Team is kept in the output for display, but the prediction is based mainly on the player's own role and production.

This does not make trades irrelevant. A new team can change minutes and usage. It simply avoids hard-coding a team name into the model.

## 5. Backtesting

The final known season, 2025-26, is used as a historical test.

The model trains on earlier season pairs and asks:

"If I only knew the older data, how accurately would I have predicted 2025-26?"

The project reports:

- MAE
- RMSE
- R-squared

It also compares the model with a simple baseline:

"next season will equal the previous season."

A model should ideally improve on that baseline.

## 6. Models

For each statistic the project compares:

- Linear Regression
- Random Forest

The model with the lower 2025-26 MAE is selected.

Separate models are trained for:

- PTS
- REB
- AST
- FG3M
- STL
- BLK

After backtesting, the selected model is retrained using all available historical pairs and applied to 2025-26 players to produce 2026-27 projections.

## 7. Current player scores

`build_rankings.py` creates simple 0-100 analytical scores.

### Scoring score
Uses:
- PPG
- true-shooting percentage
- 3PM

### Playmaking score
Uses:
- APG
- assist percentage
- turnover control

### Offense score
Combines scoring and playmaking.

### Defense score
Uses:
- steals
- blocks
- defensive rebounding percentage
- defensive rating

### All-round score
Combines offense and defense.

These are project metrics, not official NBA statistics.

## 8. Three-point score

Players need at least 2.0 3PA per game.

The score combines:
- 3PM volume
- 3P%

This avoids calling someone the best shooter because they made a very high percentage on almost no attempts.

## 9. Mid-range score

The NBA shot-location endpoint gives:
- mid-range attempts
- mid-range makes
- mid-range FG%

Players need at least 1.5 mid-range attempts per game.

The score combines:
- efficiency
- volume

## 10. SQL

`build_database.py` loads the main outputs into SQLite.

Open:

`database/nba_predictions.db`

Then run the queries in:

`sql/analysis_queries.sql`

## 11. Power BI

The files in `outputs/` are already shaped for visualization.

The suggested dashboard layout is in:

`powerbi/README.md`

## Limitations

This project predicts performance conditional on a player actually playing in 2026-27.

It cannot perfectly know:
- future injuries
- retirement
- sudden rotation changes
- trades that dramatically change a role
- rookie NBA performance with no prior NBA season

Those are real limitations and should be mentioned when presenting the model.

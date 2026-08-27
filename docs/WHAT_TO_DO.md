# What you need to do

## Mac setup

First check Python:

```bash
python3 --version
```

Use Python 3.10 or newer. Python 3.11 is a good choice for this project.

Then from Terminal:

```bash
cd NBA2627PlayerPredictions

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -r requirements.txt
```

## First full run

```bash
python3 src/run_all.py
```

The first run downloads the historical NBA data, trains the models and creates the outputs.

NBA Stats occasionally times out. If a download request fails, run the same command again. Completed seasons are skipped.

## Re-run without downloading

Once all raw data exists:

```bash
python3 src/run_all.py --skip-download
```

## Check the outputs

Open:

```text
outputs/predictions_2026_27.csv
outputs/model_metrics.csv
outputs/player_scores_2025_26.csv
outputs/leaderboards_2026_27.csv
```

## Check SQL

Open the database:

```bash
sqlite3 database/nba_predictions.db
```

Then:

```sql
.tables
```

To run all example queries from Terminal:

```bash
sqlite3 -header -column database/nba_predictions.db < sql/analysis_queries.sql
```

## Build Power BI

Follow:

```text
powerbi/README.md
```

## Before putting it on your resume

Make sure you can explain:

1. why `PLAYER_ID` is used
2. why team is not a prediction feature
3. what a player-season row means
4. why the split is chronological
5. what MAE means
6. why a previous-season baseline matters
7. why the project has separate models for each statistic
8. what the custom offense and defense scores mean
9. the limitations around injuries and rookies

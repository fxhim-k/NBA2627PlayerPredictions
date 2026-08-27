import numpy as np
import pandas as pd

from config import (
    LATEST_SEASON,
    MIN_GAMES_MODEL,
    PROCESSED,
    TARGETS,
)

CURRENT_FEATURES = [
    "AGE",
    "GP",
    "MIN",
    "FGA",
    "FG_PCT",
    "FG3A",
    "FG3_PCT",
    "FTA",
    "FT_PCT",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PTS",
    "PLUS_MINUS",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "DREB_PCT",
    "REB_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PIE",
]

CHANGE_COLUMNS = [
    "MIN",
    "PTS",
    "REB",
    "AST",
    "FG3M",
    "STL",
    "BLK",
    "USG_PCT",
    "TS_PCT",
]


def add_change_features(df):
    df = df.copy()
    grouped = df.groupby("PLAYER_ID", group_keys=False)

    previous_season = grouped["SEASON_END"].shift(1)
    consecutive = previous_season.eq(df["SEASON_END"] - 1)

    for column in CHANGE_COLUMNS:
        if column not in df.columns:
            continue

        previous_value = grouped[column].shift(1)
        change = df[column] - previous_value
        df[f"{column}_CHANGE"] = change.where(consecutive, 0).fillna(0)

    return df


def add_targets(df):
    df = df.copy()
    grouped = df.groupby("PLAYER_ID", group_keys=False)

    next_season = grouped["SEASON_END"].shift(-1)
    df["TARGET_SEASON_END"] = next_season
    df["HAS_NEXT_SEASON"] = next_season.eq(df["SEASON_END"] + 1)

    for target in TARGETS:
        df[f"TARGET_{target}"] = grouped[target].shift(-1)

    return df


def main():
    history = pd.read_csv(PROCESSED / "player_history.csv")
    history = history.sort_values(
        ["PLAYER_ID", "SEASON_END"]
    ).reset_index(drop=True)

    features = add_change_features(history)
    features = add_targets(features)

    model_rows = features[
        features["HAS_NEXT_SEASON"]
        & (features["GP"] >= MIN_GAMES_MODEL)
    ].copy()

    current_rows = features[
        features["SEASON"].eq(LATEST_SEASON)
    ].copy()

    model_file = PROCESSED / "model_dataset.csv"
    current_file = PROCESSED / "current_features.csv"

    model_rows.to_csv(model_file, index=False)
    current_rows.to_csv(current_file, index=False)

    print(f"Training rows: {len(model_rows):,}")
    print(f"Current players: {len(current_rows):,}")
    print(f"Output: {model_file}")
    print(f"Output: {current_file}")


if __name__ == "__main__":
    main()

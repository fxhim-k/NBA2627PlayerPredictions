import pandas as pd

from config import (
    PROCESSED,
    RAW_ADVANCED,
    RAW_BASE,
    SEASONS,
)

BASE_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "AGE",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PTS",
    "PLUS_MINUS",
]

ADVANCED_COLUMNS = [
    "PLAYER_ID",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "AST_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "EFG_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE",
]


def available_columns(df, wanted):
    return [column for column in wanted if column in df.columns]


def load_season(season):
    base = pd.read_csv(RAW_BASE / f"{season}.csv")
    advanced = pd.read_csv(RAW_ADVANCED / f"{season}.csv")

    base = base[available_columns(base, BASE_COLUMNS)].copy()
    advanced = advanced[available_columns(advanced, ADVANCED_COLUMNS)].copy()

    # LeagueDashPlayerStats should already be one row per player.
    # This keeps the row with the most games if a duplicate appears.
    base = (
        base.sort_values("GP", ascending=False)
        .drop_duplicates("PLAYER_ID")
    )
    advanced = advanced.drop_duplicates("PLAYER_ID")

    df = base.merge(advanced, on="PLAYER_ID", how="left")
    df["SEASON"] = season
    df["SEASON_END"] = int(season[:4]) + 1
    return df


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    seasons = [load_season(season) for season in SEASONS]
    history = pd.concat(seasons, ignore_index=True)

    history = history.sort_values(
        ["PLAYER_ID", "SEASON_END"]
    ).reset_index(drop=True)

    output_file = PROCESSED / "player_history.csv"
    history.to_csv(output_file, index=False)

    print(f"Saved {len(history):,} player-season rows")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()

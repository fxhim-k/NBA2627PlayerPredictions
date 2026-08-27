import numpy as np
import pandas as pd

from config import (
    LATEST_SEASON,
    OUTPUTS,
    PROCESSED,
    RAW_SHOOTING,
    TARGETS,
)


def percentile(series, higher_is_better=True):
    return series.rank(
        pct=True,
        ascending=higher_is_better,
    ) * 100


def make_current_scores(current):
    df = current.copy()

    df["SCORING_SCORE"] = (
        0.55 * percentile(df["PTS"])
        + 0.25 * percentile(df["TS_PCT"])
        + 0.20 * percentile(df["FG3M"])
    )

    df["PLAYMAKING_SCORE"] = (
        0.65 * percentile(df["AST"])
        + 0.20 * percentile(df["AST_PCT"])
        + 0.15 * percentile(-df["TOV"])
    )

    df["OFFENSE_SCORE"] = (
        0.65 * df["SCORING_SCORE"]
        + 0.35 * df["PLAYMAKING_SCORE"]
    )

    df["DEFENSE_SCORE"] = (
        0.30 * percentile(df["STL"])
        + 0.30 * percentile(df["BLK"])
        + 0.20 * percentile(df["DREB_PCT"])
        + 0.20 * percentile(-df["DEF_RATING"])
    )

    df["ALL_ROUND_SCORE"] = (
        0.60 * df["OFFENSE_SCORE"]
        + 0.40 * df["DEFENSE_SCORE"]
    )

    three_point_pool = df["FG3A"].ge(2.0)
    df["THREE_POINT_SCORE"] = np.nan

    df.loc[three_point_pool, "THREE_POINT_SCORE"] = (
        0.55 * percentile(df.loc[three_point_pool, "FG3M"])
        + 0.45 * percentile(df.loc[three_point_pool, "FG3_PCT"])
    )

    score_columns = [
        "SCORING_SCORE",
        "PLAYMAKING_SCORE",
        "OFFENSE_SCORE",
        "DEFENSE_SCORE",
        "ALL_ROUND_SCORE",
        "THREE_POINT_SCORE",
    ]

    df[score_columns] = df[score_columns].round(1)
    return df


def build_current_leaderboards(scores):
    metrics = {
        "Points per game": "PTS",
        "Rebounds per game": "REB",
        "Assists per game": "AST",
        "3-pointers made per game": "FG3M",
        "Steals per game": "STL",
        "Blocks per game": "BLK",
        "Offense score": "OFFENSE_SCORE",
        "Defense score": "DEFENSE_SCORE",
        "All-round score": "ALL_ROUND_SCORE",
        "3-point shooting score": "THREE_POINT_SCORE",
    }

    rows = []

    for label, column in metrics.items():
        table = (
            scores.dropna(subset=[column])
            .sort_values(column, ascending=False)
            .head(20)
        )

        for rank, (_, player) in enumerate(table.iterrows(), 1):
            rows.append({
                "metric": label,
                "rank": rank,
                "player_name": player["PLAYER_NAME"],
                "team": player["TEAM_ABBREVIATION"],
                "value": round(player[column], 2),
                "season": LATEST_SEASON,
            })

    return pd.DataFrame(rows)


def build_projected_leaderboards(predictions):
    label_map = {
        "PTS": "Projected points per game",
        "REB": "Projected rebounds per game",
        "AST": "Projected assists per game",
        "FG3M": "Projected 3-pointers made per game",
        "STL": "Projected steals per game",
        "BLK": "Projected blocks per game",
    }

    rows = []

    for target in TARGETS:
        column = f"PROJECTED_{target}"
        table = predictions.sort_values(
            column,
            ascending=False,
        ).head(20)

        for rank, (_, player) in enumerate(table.iterrows(), 1):
            rows.append({
                "metric": label_map[target],
                "rank": rank,
                "player_name": player["PLAYER_NAME"],
                "team_2025_26": player["TEAM_ABBREVIATION"],
                "projected_value": player[column],
                "season": "2026-27",
            })

    return pd.DataFrame(rows)


def build_midrange_table():
    shot_file = RAW_SHOOTING / f"{LATEST_SEASON}.csv"

    if not shot_file.exists():
        return pd.DataFrame()

    shooting = pd.read_csv(shot_file)

    required = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "mid_range_fga",
        "mid_range_fg_pct",
    ]

    if not all(column in shooting.columns for column in required):
        return pd.DataFrame()

    table = shooting[required].copy()
    table = table[table["mid_range_fga"] >= 1.5].copy()

    table["volume_percentile"] = percentile(
        table["mid_range_fga"]
    )
    table["efficiency_percentile"] = percentile(
        table["mid_range_fg_pct"]
    )

    table["MIDRANGE_SCORE"] = (
        0.60 * table["efficiency_percentile"]
        + 0.40 * table["volume_percentile"]
    ).round(1)

    return table.sort_values(
        "MIDRANGE_SCORE",
        ascending=False,
    )


def main():
    current = pd.read_csv(PROCESSED / "player_history.csv")
    current = current[current["SEASON"] == LATEST_SEASON].copy()
    current = current[(current["GP"] >= 20) & (current["MIN"] >= 10)]

    predictions = pd.read_csv(
        OUTPUTS / "predictions_2026_27.csv"
    )

    scores = make_current_scores(current)

    scores.to_csv(
        OUTPUTS / "player_scores_2025_26.csv",
        index=False,
    )

    build_current_leaderboards(scores).to_csv(
        OUTPUTS / "leaderboards_2025_26.csv",
        index=False,
    )

    build_projected_leaderboards(predictions).to_csv(
        OUTPUTS / "leaderboards_2026_27.csv",
        index=False,
    )

    midrange = build_midrange_table()
    midrange.to_csv(
        OUTPUTS / "midrange_leaders_2025_26.csv",
        index=False,
    )

    print("Ranking files created.")
    if midrange.empty:
        print(
            "Mid-range output is empty because shot-location data "
            "was unavailable or had a different schema."
        )


if __name__ == "__main__":
    main()

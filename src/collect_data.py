import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayershotlocations,
)

from config import (
    LATEST_SEASON,
    RAW_ADVANCED,
    RAW_BASE,
    RAW_SHOOTING,
    SEASONS,
)

RETRIES = 3
PAUSE_SECONDS = 1.5


def get_player_stats(season, measure):
    request = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure,
        timeout=90,
    )
    df = request.get_data_frames()[0]

    if df.empty:
        raise ValueError(f"{season} {measure} returned no rows")

    return df


def flatten_shot_columns(df):
    columns = []

    for zone, stat in df.columns:
        zone = str(zone).strip()
        stat = str(stat).strip()

        if not zone:
            columns.append(stat)
        else:
            zone_name = (
                zone.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")
            )
            columns.append(f"{zone_name}_{stat.lower()}")

    df = df.copy()
    df.columns = columns
    return df


def get_shot_locations(season):
    request = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        distance_range="By Zone",
        measure_type_simple="Base",
        timeout=90,
    )
    df = request.get_data_frames()[0]
    return flatten_shot_columns(df)


def fetch_with_retry(fetch_function, *args):
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            return fetch_function(*args)
        except Exception as error:
            last_error = error
            print(f"  attempt {attempt} failed: {error}")
            if attempt < RETRIES:
                time.sleep(4)

    raise RuntimeError(f"NBA data request failed after {RETRIES} tries") from last_error


def save_season_data(season):
    base_file = RAW_BASE / f"{season}.csv"
    advanced_file = RAW_ADVANCED / f"{season}.csv"

    if not base_file.exists():
        print(f"{season}: base stats")
        base = fetch_with_retry(get_player_stats, season, "Base")
        base.to_csv(base_file, index=False)
        time.sleep(PAUSE_SECONDS)
    else:
        print(f"{season}: base stats already downloaded")

    if not advanced_file.exists():
        print(f"{season}: advanced stats")
        advanced = fetch_with_retry(get_player_stats, season, "Advanced")
        advanced.to_csv(advanced_file, index=False)
        time.sleep(PAUSE_SECONDS)
    else:
        print(f"{season}: advanced stats already downloaded")


def save_shot_data():
    output_file = RAW_SHOOTING / f"{LATEST_SEASON}.csv"

    if output_file.exists():
        print(f"{LATEST_SEASON}: shot locations already downloaded")
        return

    print(f"{LATEST_SEASON}: shot-location stats")
    shooting = fetch_with_retry(get_shot_locations, LATEST_SEASON)
    shooting.to_csv(output_file, index=False)


def main():
    RAW_BASE.mkdir(parents=True, exist_ok=True)
    RAW_ADVANCED.mkdir(parents=True, exist_ok=True)
    RAW_SHOOTING.mkdir(parents=True, exist_ok=True)

    print("Downloading NBA player data...")
    for season in SEASONS:
        save_season_data(season)

    print()

    try:
        save_shot_data()
    except RuntimeError as error:
        print(f"Shot-location data skipped: {error}")
        print("The main prediction pipeline can still run.")

    print("\nData collection finished.")


if __name__ == "__main__":
    main()

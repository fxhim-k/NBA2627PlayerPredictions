import sqlite3

import pandas as pd

from config import DATABASE, OUTPUTS, PROCESSED


def main():
    DATABASE.mkdir(parents=True, exist_ok=True)
    db_file = DATABASE / "nba_predictions.db"

    tables = {
        "player_history": PROCESSED / "player_history.csv",
        "predictions_2026_27": OUTPUTS / "predictions_2026_27.csv",
        "player_scores_2025_26": OUTPUTS / "player_scores_2025_26.csv",
        "model_metrics": OUTPUTS / "model_metrics.csv",
    }

    midrange_file = OUTPUTS / "midrange_leaders_2025_26.csv"
    if midrange_file.exists() and midrange_file.stat().st_size > 1:
        tables["midrange_leaders_2025_26"] = midrange_file

    with sqlite3.connect(db_file) as connection:
        for table_name, csv_file in tables.items():
            df = pd.read_csv(csv_file)
            df.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False,
            )
            print(f"{table_name}: {len(df):,} rows")

    print(f"\nSQLite database: {db_file}")


if __name__ == "__main__":
    main()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

from config import (
    MIN_GAMES_CURRENT,
    MODELS,
    OUTPUTS,
    PREDICTION_SEASON,
    PROCESSED,
    TARGETS,
)

TEST_SEASON_END = 2026

FEATURES = [
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
    "MIN_CHANGE",
    "PTS_CHANGE",
    "REB_CHANGE",
    "AST_CHANGE",
    "FG3M_CHANGE",
    "STL_CHANGE",
    "BLK_CHANGE",
    "USG_PCT_CHANGE",
    "TS_PCT_CHANGE",
]

BASELINE_COLUMN = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",
    "STL": "STL",
    "BLK": "BLK",
}


def usable_features(df):
    return [column for column in FEATURES if column in df.columns]


def make_models():
    return {
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def score_model(model, x_test, y_test):
    predictions = model.predict(x_test)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }


def train_target(df, current, target, feature_columns):
    target_column = f"TARGET_{target}"

    train = df[df["TARGET_SEASON_END"] < TEST_SEASON_END].copy()
    test = df[df["TARGET_SEASON_END"] == TEST_SEASON_END].copy()

    train = train.dropna(subset=[target_column])
    test = test.dropna(subset=[target_column])

    x_train = train[feature_columns]
    y_train = train[target_column]
    x_test = test[feature_columns]
    y_test = test[target_column]

    metrics = []

    baseline = test[BASELINE_COLUMN[target]]
    metrics.append({
        "target": target,
        "model": "Previous Season Baseline",
        "mae": mean_absolute_error(y_test, baseline),
        "rmse": mean_squared_error(y_test, baseline) ** 0.5,
        "r2": r2_score(y_test, baseline),
    })

    fitted_models = {}

    for name, model in make_models().items():
        model.fit(x_train, y_train)
        fitted_models[name] = model

        scores = score_model(model, x_test, y_test)
        metrics.append({
            "target": target,
            "model": name,
            **scores,
        })

    model_metrics = pd.DataFrame(metrics)
    candidates = model_metrics[
        model_metrics["model"] != "Previous Season Baseline"
    ]
    best_name = candidates.sort_values("mae").iloc[0]["model"]

    final_model = make_models()[best_name]
    full_rows = df.dropna(subset=[target_column])
    final_model.fit(
        full_rows[feature_columns],
        full_rows[target_column],
    )

    joblib.dump(final_model, MODELS / f"{target.lower()}_model.joblib")

    current_prediction = final_model.predict(
        current[feature_columns]
    )

    return current_prediction, model_metrics, best_name


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(PROCESSED / "model_dataset.csv")
    current = pd.read_csv(PROCESSED / "current_features.csv")

    current = current[
        (current["GP"] >= MIN_GAMES_CURRENT)
        & (current["MIN"] >= 10)
    ].copy()

    feature_columns = usable_features(dataset)

    predictions = current[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "AGE",
            "GP",
            "MIN",
        ]
    ].copy()

    predictions = predictions.rename(columns={
        "AGE": "AGE_2025_26",
        "GP": "GP_2025_26",
        "MIN": "MIN_2025_26",
    })
    predictions["AGE_2026_27"] = predictions["AGE_2025_26"] + 1

    all_metrics = []
    selected_models = []

    for target in TARGETS:
        projected, metrics, best_name = train_target(
            dataset,
            current,
            target,
            feature_columns,
        )

        predictions[f"PROJECTED_{target}"] = np.maximum(
            projected,
            0,
        ).round(2)

        metrics["test_season"] = "2025-26"
        all_metrics.append(metrics)

        selected_models.append({
            "target": target,
            "selected_model": best_name,
        })

        print(f"{target}: {best_name}")

    predictions["PREDICTION_SEASON"] = PREDICTION_SEASON

    predictions.to_csv(
        OUTPUTS / "predictions_2026_27.csv",
        index=False,
    )

    pd.concat(all_metrics, ignore_index=True).to_csv(
        OUTPUTS / "model_metrics.csv",
        index=False,
    )

    pd.DataFrame(selected_models).to_csv(
        OUTPUTS / "selected_models.csv",
        index=False,
    )

    print(f"\nPredicted {len(predictions):,} returning-player seasons")
    print(f"Output: {OUTPUTS / 'predictions_2026_27.csv'}")


if __name__ == "__main__":
    main()

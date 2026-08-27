from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_BASE = ROOT / "data" / "raw" / "base"
RAW_ADVANCED = ROOT / "data" / "raw" / "advanced"
RAW_SHOOTING = ROOT / "data" / "raw" / "shooting"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DATABASE = ROOT / "database"

SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

LATEST_SEASON = "2025-26"
PREDICTION_SEASON = "2026-27"

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

MIN_GAMES_MODEL = 15
MIN_GAMES_CURRENT = 20

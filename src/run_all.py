import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_STEP = "src/collect_data.py"

STEPS = [
    "src/prepare_data.py",
    "src/build_features.py",
    "src/train_models.py",
    "src/build_rankings.py",
    "src/build_database.py",
]


def run(script):
    print(f"\n--- {script} ---")
    subprocess.run(
        [sys.executable, str(ROOT / script)],
        check=True,
    )


def main():
    skip_download = "--skip-download" in sys.argv

    if not skip_download:
        run(DOWNLOAD_STEP)

    for step in STEPS:
        run(step)

    print("\nNBA2627PlayerPredictions finished.")


if __name__ == "__main__":
    main()

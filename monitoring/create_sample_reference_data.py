from pathlib import Path

import pandas as pd


INPUT_PATH = Path("tests/fixtures/sample_restaurants.csv")
OUTPUT_PATH = Path("data/production_logs/reference.parquet")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference_data = pd.read_csv(INPUT_PATH)
    reference_data.to_parquet(OUTPUT_PATH, index=False)

    print(f"Sample reference data written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
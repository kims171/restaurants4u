import json
from pathlib import Path

import pandas as pd
import yaml


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()["validate_data"]

    raw_path = Path(config["raw_input_path"])
    output_path = Path(config["validated_output_path"])
    report_path = Path(config["validation_report_path"])

    rating_col = config["rating_column"]
    lat_col = config["latitude_column"]
    lon_col = config["longitude_column"]

    df = pd.read_csv(raw_path)

    original_rows = len(df)

    report = {
        "original_rows": int(original_rows),
        "original_columns": int(df.shape[1]),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in [rating_col, lat_col, lon_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report["invalid_rating_count"] = int(df[rating_col].isna().sum())
    report["invalid_latitude_count"] = int(df[lat_col].isna().sum())
    report["invalid_longitude_count"] = int(df[lon_col].isna().sum())

    report["rating_out_of_range_count"] = int(
        (~df[rating_col].between(config["min_rating"], config["max_rating"])).sum()
    )
    report["latitude_out_of_range_count"] = int((~df[lat_col].between(-90, 90)).sum())
    report["longitude_out_of_range_count"] = int((~df[lon_col].between(-180, 180)).sum())

    # Remove invalid records
    df = df.dropna(subset=[rating_col, lat_col, lon_col])
    df = df[df[rating_col].between(config["min_rating"], config["max_rating"])]
    df = df[df[lat_col].between(-90, 90)]
    df = df[df[lon_col].between(-180, 180)]

    # Drop duplicate restaurants if useful columns exist
    duplicate_subset = [col for col in ["Title", "Address"] if col in df.columns]
    if duplicate_subset:
        duplicate_business_count = int(df.duplicated(subset=duplicate_subset).sum())
        df = df.drop_duplicates(subset=duplicate_subset)
    else:
        duplicate_business_count = 0

    report["duplicate_business_count"] = duplicate_business_count
    report["validated_rows"] = int(len(df))
    report["rows_removed"] = int(original_rows - len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Data validation completed.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
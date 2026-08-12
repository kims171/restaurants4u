"""Data drift detection using EvidentlyAI.

Compares the reference distribution (training data, from
data/processed/train.parquet) against a window of recent production
inference requests (current data) to decide whether the model needs
retraining.

Usage:
    python monitoring/drift_detection.py \
        --reference data/processed/train.parquet \
        --current data/production_logs/latest.parquet \
        --report-out monitoring/reports/drift_report.html \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "has_website",
    "has_phone",
    "has_images",
    "category",
]


def load_data(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df[columns]


def run_drift_check(
    reference_path: Path,
    current_path: Path,
    report_out: Path,
    threshold: float,
) -> dict:
    reference = load_data(reference_path, FEATURE_COLUMNS)
    current = load_data(current_path, FEATURE_COLUMNS)

    column_mapping = ColumnMapping(
        numerical_features=["latitude", "longitude"],
        categorical_features=["has_website", "has_phone", "has_images", "category"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_out))

    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    share_drifted = drift_metric["share_of_drifted_columns"]
    dataset_drift = drift_metric["dataset_drift"]

    summary = {
        "share_of_drifted_columns": share_drifted,
        "dataset_drift_detected": dataset_drift,
        "retrain_recommended": share_drifted >= threshold,
        "threshold": threshold,
        "report_path": str(report_out),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("data/processed/train.parquet"))
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--report-out", type=Path, default=Path("monitoring/reports/drift_report.html")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Share of drifted columns (0-1) above which retraining is recommended",
    )
    parser.add_argument("--summary-out", type=Path, default=Path("monitoring/drift_summary.json"))
    args = parser.parse_args()

    summary = run_drift_check(args.reference, args.current, args.report_out, args.threshold)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    # Exit code 1 signals "retrain recommended" to the calling workflow step,
    # without failing the job outright (workflow checks this explicitly).
    if summary["retrain_recommended"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

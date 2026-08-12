import json
import os

import pandas as pd
import yaml

from features import engineer_features  # noqa: F401  (re-exported for callers/tests)


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    feat_cfg = config["feature_engineering"]

    processed_dir = feat_cfg["processed_dir"]
    features_dir = feat_cfg["features_dir"]

    os.makedirs(features_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    validation_df = pd.read_parquet(os.path.join(processed_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))

    train_features, top_categories = engineer_features(
        train_df,
        top_categories=None,
        top_category_count=feat_cfg["top_category_count"],
    )

    validation_features, _ = engineer_features(
        validation_df,
        top_categories=top_categories,
        top_category_count=feat_cfg["top_category_count"],
    )

    test_features, _ = engineer_features(
        test_df,
        top_categories=top_categories,
        top_category_count=feat_cfg["top_category_count"],
    )

    # Align validation/test with train columns
    validation_features = validation_features.reindex(columns=train_features.columns, fill_value=0)
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

    train_features.to_parquet(os.path.join(features_dir, "train_features.parquet"), index=False)
    validation_features.to_parquet(os.path.join(features_dir, "validation_features.parquet"), index=False)
    test_features.to_parquet(os.path.join(features_dir, "test_features.parquet"), index=False)

    report = {
        "train_shape": list(train_features.shape),
        "validation_shape": list(validation_features.shape),
        "test_shape": list(test_features.shape),
        "top_categories": top_categories,
        "feature_columns": train_features.columns.tolist(),
    }

    os.makedirs(os.path.dirname(feat_cfg["feature_report_path"]), exist_ok=True)

    with open(feat_cfg["feature_report_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Feature engineering completed.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()

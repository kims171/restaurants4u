import json
import os

import pandas as pd
import yaml


TARGET = "is_highly_rated"


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def engineer_features(df, top_categories=None, top_category_count=20):
    df = df.copy()

    # Presence-based features
    for source_col, new_col in [
        ("Website", "has_website"),
        ("Phone", "has_phone"),
        ("Images", "has_images"),
    ]:
        if source_col in df.columns:
            df[new_col] = df[source_col].notna().astype(int)
        else:
            df[new_col] = 0

    # Text-length structural feature
    if "Title" in df.columns:
        df["title_length"] = df["Title"].fillna("").astype(str).str.len()
    else:
        df["title_length"] = 0

    if "Address" in df.columns:
        df["address_length"] = df["Address"].fillna("").astype(str).str.len()
    else:
        df["address_length"] = 0

    # Category capping
    if "Category" in df.columns:
        df["Category"] = df["Category"].fillna("Unknown").astype(str)

        if top_categories is None:
            top_categories = df["Category"].value_counts().head(top_category_count).index.tolist()

        df["clean_category"] = df["Category"].apply(
            lambda x: x if x in top_categories else "Other"
        )
    else:
        df["clean_category"] = "Unknown"
        if top_categories is None:
            top_categories = ["Unknown"]

    category_encoded = pd.get_dummies(df["clean_category"], prefix="cat")

    numeric_columns = []

    for col in ["Latitude", "Longitude", "has_website", "has_phone", "has_images", "title_length", "address_length"]:
        if col in df.columns:
            numeric_columns.append(col)

    feature_df = pd.concat(
        [df[numeric_columns], category_encoded, df[[TARGET]]],
        axis=1,
    )

    return feature_df, top_categories


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
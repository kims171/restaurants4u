import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data():
    full_config = load_config()
    config = full_config["prepare"]

    df = pd.read_parquet(config["validated_input_path"])

    rating_col = full_config["validate_data"]["rating_column"]

    df["is_highly_rated"] = (
        df[rating_col] >= config["positive_rating_threshold"]
    ).astype(int)

    # Do not allow leakage from the original rating column
    df = df.drop(columns=[rating_col])

    train_df, temp_df = train_test_split(
        df,
        test_size=config["test_size"] + config["validation_size"],
        random_state=config["seed"],
        stratify=df["is_highly_rated"],
    )

    relative_test_size = config["test_size"] / (
        config["test_size"] + config["validation_size"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=config["seed"],
        stratify=temp_df["is_highly_rated"],
    )

    os.makedirs(config["processed_dir"], exist_ok=True)

    train_path = os.path.join(config["processed_dir"], "train.parquet")
    validation_path = os.path.join(config["processed_dir"], "validation.parquet")
    test_path = os.path.join(config["processed_dir"], "test.parquet")

    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("Prepared train/validation/test splits.")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test: {test_df.shape}")


if __name__ == "__main__":
    prepare_data()
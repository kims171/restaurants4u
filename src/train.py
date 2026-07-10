import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET = "is_highly_rated"


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model():
    config = load_config()
    train_cfg = config["train"]
    feature_cfg = config["feature_engineering"]

    mlflow.set_tracking_uri(train_cfg["tracking_uri"])
    mlflow.set_experiment(train_cfg["experiment_name"])

    train_df = pd.read_parquet(
        os.path.join(feature_cfg["features_dir"], "train_features.parquet")
    )
    validation_df = pd.read_parquet(
        os.path.join(feature_cfg["features_dir"], "validation_features.parquet")
    )

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_val = validation_df.drop(columns=[TARGET])
    y_val = validation_df[TARGET]

    with mlflow.start_run(run_name=train_cfg["run_name"]):
        mlflow.log_params(
            {
                "model_type": train_cfg["model_type"],
                "n_estimators": train_cfg["n_estimators"],
                "max_depth": train_cfg["max_depth"],
                "min_samples_split": train_cfg["min_samples_split"],
                "class_weight": train_cfg["class_weight"],
                "feature_count": X_train.shape[1],
                "train_rows": X_train.shape[0],
                "validation_rows": X_val.shape[0],
            }
        )

        clf = RandomForestClassifier(
            n_estimators=train_cfg["n_estimators"],
            max_depth=train_cfg["max_depth"],
            min_samples_split=train_cfg["min_samples_split"],
            class_weight=train_cfg["class_weight"],
            random_state=config["prepare"]["seed"],
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]

        metrics = {
            "validation_accuracy": accuracy_score(y_val, val_preds),
            "validation_balanced_accuracy": balanced_accuracy_score(y_val, val_preds),
            "validation_precision": precision_score(y_val, val_preds, zero_division=0),
            "validation_recall": recall_score(y_val, val_preds, zero_division=0),
            "validation_f1": f1_score(y_val, val_preds, zero_division=0),
            "validation_roc_auc": roc_auc_score(y_val, val_probs),
        }

        mlflow.log_metrics(metrics)

        os.makedirs(os.path.dirname(train_cfg["model_path"]), exist_ok=True)
        joblib.dump(clf, train_cfg["model_path"])

        mlflow.sklearn.log_model(clf, "model")

        print("Model trained and logged to MLflow.")
        print(metrics)


if __name__ == "__main__":
    train_model()
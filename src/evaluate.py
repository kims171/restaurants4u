import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


TARGET = "is_highly_rated"


def load_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model():
    config = load_config()
    train_cfg = config["train"]
    eval_cfg = config["evaluate"]
    feature_cfg = config["feature_engineering"]

    mlflow.set_tracking_uri(train_cfg["tracking_uri"])
    mlflow.set_experiment(train_cfg["experiment_name"])

    model = joblib.load(train_cfg["model_path"])

    test_df = pd.read_parquet(
        os.path.join(feature_cfg["features_dir"], "test_features.parquet")
    )

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds, zero_division=0),
        "test_recall": recall_score(y_test, preds, zero_division=0),
        "test_f1": f1_score(y_test, preds, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probs),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    os.makedirs("reports", exist_ok=True)

    with open(eval_cfg["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    with open(eval_cfg["classification_report_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    matrix_display = ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(eval_cfg["confusion_matrix_path"], dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probs)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metrics["test_roc_auc"]).plot()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(eval_cfg["roc_curve_path"], dpi=200)
    plt.close()

    precision_values, recall_values, _ = precision_recall_curve(y_test, probs)
    PrecisionRecallDisplay(
        precision=precision_values,
        recall=recall_values,
    ).plot()
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(eval_cfg["precision_recall_curve_path"], dpi=200)
    plt.close()

    with mlflow.start_run(run_name=f"{train_cfg['run_name']}_test_evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(eval_cfg["metrics_path"])
        mlflow.log_artifact(eval_cfg["classification_report_path"])
        mlflow.log_artifact(eval_cfg["confusion_matrix_path"])
        mlflow.log_artifact(eval_cfg["roc_curve_path"])
        mlflow.log_artifact(eval_cfg["precision_recall_curve_path"])

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    evaluate_model()
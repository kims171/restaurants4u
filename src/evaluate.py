import os
import json
import yaml
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_model():
    config = load_config()
    
    model = joblib.load(config["train"]["model_path"])
    test_df = pd.read_parquet(os.path.join(config["prepare"]["processed_dir"], "test.parquet"))
    
    X_test = test_df.drop(columns=["is_highly_rated"])
    y_test = test_df["is_highly_rated"]
    
    preds = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0)
    }
    
    with open(config["evaluate"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics exported to JSON file: {metrics}")

if __name__ == "__main__":
    evaluate_model()
import os
import yaml
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    train_cfg = config["train"]

    # Establish centralized connection properties
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Restaurant_Metadata_Ranking_Classifier")

    train_df = pd.read_parquet(os.path.join(config["prepare"]["processed_dir"], "train.parquet"))
    X_train = train_df.drop(columns=["is_highly_rated"])
    y_train = train_df["is_highly_rated"]

    with mlflow.start_run():
        # Log parameter configs inside active context block
        mlflow.log_params(train_cfg)

        clf = RandomForestClassifier(
            n_estimators=train_cfg["n_estimators"],
            max_depth=train_cfg["max_depth"],
            random_state=config["prepare"]["seed"]
        )
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_acc)

        os.makedirs(os.path.dirname(train_cfg["model_path"]), exist_ok=True)
        joblib.dump(clf, train_cfg["model_path"])
        mlflow.sklearn.log_model(clf, "model")
        print(f"Model serialized successfully. Internal Accuracy: {train_acc:.4f}")

if __name__ == "__main__":
    train_model()
"""DVC pipeline stage 3: TRAIN.

Trains every candidate model listed in params.yaml and logs each as its own
MLflow run under one experiment. This is the heart of the "data scientist /
MLOps engineer" hand-off you described: instead of one data scientist
picking "the" model by eye, we train several candidates the same way, on
the same data, and let MLflow hold the full comparable record (params,
metrics, the model artifact itself) so the *next* stage can pick a winner
programmatically instead of manually.

Why MLflow here and not just print()/a CSV: MLflow's tracking server keeps
every run queryable later (by whoever does the model-registry / deployment
step), and mlflow.sklearn.log_model() packages the model with its exact
library versions - so "the model that won" is never ambiguous.

Run: python -m src.models.train
"""
import json
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MLFLOW_TRACKING_URI

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
METRICS_PATH = "reports/train_metrics.json"
TARGET = "med_house_val"

MODEL_BUILDERS = {
    "linear_regression": lambda p: Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression(**p))]
    ),
    "random_forest": lambda p: Pipeline(
        [("model", RandomForestRegressor(**p))]
    ),
    "gradient_boosting": lambda p: Pipeline(
        [("model", GradientBoostingRegressor(**p))]
    ),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "no-commit-yet"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    commit = git_commit()
    all_metrics = {}

    for model_name, model_params in params["models"].items():
        model_params = model_params or {}
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags({"git_commit": commit, "stage": "train"})
            mlflow.log_params({f"{model_name}.{k}": v for k, v in model_params.items()})
            mlflow.log_param("n_train_rows", len(X_train))

            pipeline = MODEL_BUILDERS[model_name](model_params)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            all_metrics[model_name] = {
                "run_id": run.info.run_id,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
            print(f"[{model_name}] rmse={rmse:.4f} mae={mae:.4f} r2={r2:.4f} (run_id={run.info.run_id})")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()

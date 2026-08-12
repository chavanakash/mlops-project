"""DVC pipeline stage 4: SELECT BEST MODEL (the MLOps engineer's step).

Reads the metrics every candidate model produced in train.py, picks the one
with the lowest test-set RMSE, and registers *that specific run's model
artifact* into the MLflow Model Registry under a stable name
(`house-price-model`). We then point the "champion" alias at it.

Why a registry + alias instead of just "the best file on disk": the
registry is what the serving app and deployment pipeline will read from
later. Serving code asks for "give me the model tagged champion" - it never
needs to know which algorithm won, or its run_id. That indirection is what
lets you swap in a better model later (a new challenger beats champion)
without touching a single line of serving/deployment code - you just move
the alias.

Run: python -m src.models.select_best_model
"""
import json

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI

METRICS_PATH = "reports/train_metrics.json"
BEST_MODEL_PATH = "reports/best_model.json"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    registered_model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    with open(METRICS_PATH) as f:
        all_metrics = json.load(f)

    best_model_name, best = min(all_metrics.items(), key=lambda kv: kv[1]["rmse"])
    best_run_id = best["run_id"]

    print(f"Best model: {best_model_name} (run_id={best_run_id}, rmse={best['rmse']:.4f})")

    try:
        client.create_registered_model(registered_model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    model_version = client.create_model_version(
        name=registered_model_name,
        source=f"runs:/{best_run_id}/model",
        run_id=best_run_id,
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=model_version.version,
    )

    result = {
        "registered_model_name": registered_model_name,
        "version": model_version.version,
        "algorithm": best_model_name,
        "run_id": best_run_id,
        "rmse": best["rmse"],
        "mae": best["mae"],
        "r2": best["r2"],
        "alias": "champion",
    }
    with open(BEST_MODEL_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Registered {registered_model_name} v{model_version.version} -> alias 'champion'")


if __name__ == "__main__":
    main()

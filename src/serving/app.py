"""Prediction API - the DevOps/deployment hand-off point.

Why this app never hardcodes "which algorithm": it asks the MLflow Model
Registry for whichever model version currently holds the `champion` alias
(set by src/models/select_best_model.py). When a future retraining run
promotes a better challenger to champion, this app can pick it up via
POST /admin/reload - no code change, no redeploy required. That indirection
is the entire point of a model registry in production MLOps.

Why every request gets logged to the `predictions` table: house sale
prices aren't known instantly. We record the prediction now; once a sale
closes we can backfill `actual_value` and measure real-world model error
over time (see src/monitoring/). Without this log there is nothing to
compare "what we predicted" against later.

Run: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, MLFLOW_TRACKING_URI

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Beyond the generic request-rate/latency metrics Instrumentator adds below,
# this tracks the model's own output distribution over time - a sudden
# shift here (predictions clustering at a different price range than
# usual) is often the first hint of data drift, visible in Grafana well
# before anyone backfills actual_value to compute real error.
PREDICTED_VALUE = Histogram(
    "house_price_predicted_value",
    "Distribution of predicted house values ($100k units)",
    buckets=(0.5, 1, 1.5, 2, 3, 4, 5, 7, 10),
)

with open("params.yaml") as f:
    PARAMS = yaml.safe_load(f)
REGISTERED_MODEL_NAME = PARAMS["mlflow"]["registered_model_name"]

engine = create_engine(DATABASE_URL)
model_state: dict = {"model": None, "version": None}


def load_champion() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    model_state["model"] = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL_NAME}@champion"
    )
    model_state["version"] = mv.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion()
    yield


app = FastAPI(title="House Price Prediction API", lifespan=lifespan)

# Exposes GET /metrics (request count/latency/status by endpoint) for
# Prometheus to scrape - see docker/prometheus.yml's house-price-app job.
Instrumentator().instrument(app).expose(app)


@app.get("/", include_in_schema=False)
def frontend():
    """A plain HTML/JS form - easier to eyeball predictions than Swagger UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class HouseFeatures(BaseModel):
    med_inc: float = Field(..., description="Median income in the block group (10k USD)")
    house_age: float = Field(..., ge=0)
    ave_rooms: float = Field(..., gt=0)
    ave_bedrms: float = Field(..., gt=0)
    population: float = Field(..., ge=0)
    ave_occup: float = Field(..., gt=0)
    latitude: float
    longitude: float


class PredictionResponse(BaseModel):
    predicted_value: float
    model_name: str
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_state["version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = features.model_dump()
    row["bedrms_ratio"] = row["ave_bedrms"] / row["ave_rooms"]
    X = pd.DataFrame([row])

    prediction = float(model_state["model"].predict(X)[0])
    PREDICTED_VALUE.observe(prediction)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions
                    (med_inc, house_age, ave_rooms, ave_bedrms, population,
                     ave_occup, latitude, longitude, bedrms_ratio,
                     predicted_value, model_name, model_version)
                VALUES
                    (:med_inc, :house_age, :ave_rooms, :ave_bedrms, :population,
                     :ave_occup, :latitude, :longitude, :bedrms_ratio,
                     :predicted_value, :model_name, :model_version)
                """
            ),
            {
                **row,
                "predicted_value": prediction,
                "model_name": REGISTERED_MODEL_NAME,
                "model_version": str(model_state["version"]),
            },
        )

    return PredictionResponse(
        predicted_value=prediction,
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(model_state["version"]),
    )


@app.post("/admin/reload")
def reload_model():
    """Re-fetch whichever model version currently holds the champion alias.

    Call this after a retraining run promotes a new challenger to champion,
    so the running API picks it up without a restart/redeploy.
    """
    load_champion()
    return {"reloaded": True, "model_version": model_state["version"]}

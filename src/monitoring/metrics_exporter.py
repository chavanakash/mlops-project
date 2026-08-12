"""Background exporter: periodically recomputes drift + live model
performance metrics and exposes them on GET /metrics for Prometheus - the
same mechanism the serving app uses for its own request metrics, but run
as a separate process. Reference distribution is the warehouse's
historical_batch rows (what the model was trained on); current
distribution is the most recent prediction requests (what the model is
actually seeing in production right now).

Running this as its own process (rather than doing the computation inside
the serving app) means a slow drift calculation can never add latency to
an actual prediction request.

Run: python -m src.monitoring.metrics_exporter
"""
import time

import numpy as np
import pandas as pd
from prometheus_client import Gauge, start_http_server
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL
from src.monitoring.drift import ks_drift, population_stability_index

FEATURE_COLUMNS = [
    "med_inc", "house_age", "ave_rooms", "ave_bedrms",
    "population", "ave_occup", "latitude", "longitude", "bedrms_ratio",
]

CHECK_INTERVAL_SECONDS = 30
CURRENT_WINDOW_ROWS = 500

KS_PVALUE = Gauge(
    "feature_drift_ks_pvalue",
    "KS-test p-value: live prediction-request features vs historical training data (low = drift)",
    ["feature"],
)
PSI = Gauge(
    "feature_drift_psi",
    "Population Stability Index: live prediction-request features vs historical training data",
    ["feature"],
)
PERF_RMSE = Gauge("model_performance_rmse", "RMSE over predictions with a backfilled actual_value")
PERF_MAE = Gauge("model_performance_mae", "MAE over predictions with a backfilled actual_value")
LABELED_COUNT = Gauge(
    "model_performance_labeled_count", "Number of predictions with a backfilled actual_value"
)
PREDICTIONS_COUNT = Gauge("predictions_total_count", "Prediction requests considered this refresh")


def load_reference(engine) -> pd.DataFrame:
    df = pd.read_sql(text("SELECT * FROM houses WHERE source = 'historical_batch'"), engine)
    df["bedrms_ratio"] = df["ave_bedrms"] / df["ave_rooms"]
    return df


def load_current(engine, lookback_rows: int = CURRENT_WINDOW_ROWS) -> pd.DataFrame:
    query = text("SELECT * FROM predictions ORDER BY requested_at DESC LIMIT :n")
    return pd.read_sql(query, engine, params={"n": lookback_rows})


def update_drift_metrics(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    for col in FEATURE_COLUMNS:
        if current[col].dropna().shape[0] < 2:
            continue  # not enough live samples yet for a meaningful test
        try:
            _, p_value = ks_drift(reference[col], current[col])
            psi = population_stability_index(reference[col], current[col])
        except Exception as e:
            print(f"metrics_exporter: drift calc failed for {col}: {e}")
            continue
        KS_PVALUE.labels(feature=col).set(p_value)
        PSI.labels(feature=col).set(psi)


def update_performance_metrics(current: pd.DataFrame) -> None:
    labeled = current.dropna(subset=["actual_value"])
    LABELED_COUNT.set(len(labeled))
    if labeled.empty:
        return
    errors = labeled["predicted_value"] - labeled["actual_value"]
    PERF_RMSE.set(float(np.sqrt((errors**2).mean())))
    PERF_MAE.set(float(errors.abs().mean()))


def main() -> None:
    engine = create_engine(DATABASE_URL)
    start_http_server(8001)
    print("Metrics exporter listening on :8001/metrics")

    while True:
        try:
            reference = load_reference(engine)
            current = load_current(engine)
            PREDICTIONS_COUNT.set(len(current))
            if not current.empty:
                update_drift_metrics(reference, current)
                update_performance_metrics(current)
        except Exception as e:
            print(f"metrics_exporter: error during refresh: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

"""Tests the serving API's request/response contract without touching real
MLflow or Postgres - CI has neither, and shouldn't need them just to check
that /predict validates input and returns what the model produced.
"""
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from src.serving import app as app_module


class DummyModel:
    def predict(self, X):
        return [3.14] * len(X)


def test_predict_returns_prediction(monkeypatch):
    monkeypatch.setattr(app_module, "load_champion", lambda: None)
    app_module.model_state["model"] = DummyModel()
    app_module.model_state["version"] = "test"

    # StaticPool keeps one connection alive for the engine's lifetime -
    # a plain in-memory sqlite engine hands out a fresh (empty) database
    # per connection, which would make the table we create below invisible
    # to the app's own engine.begin() calls.
    test_engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    with test_engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    med_inc REAL, house_age REAL, ave_rooms REAL, ave_bedrms REAL,
                    population REAL, ave_occup REAL, latitude REAL, longitude REAL,
                    bedrms_ratio REAL, predicted_value REAL,
                    model_name TEXT, model_version TEXT
                )
                """
            )
        )
    monkeypatch.setattr(app_module, "engine", test_engine)

    with TestClient(app_module.app) as client:
        resp = client.post(
            "/predict",
            json={
                "med_inc": 5.0,
                "house_age": 10,
                "ave_rooms": 6.0,
                "ave_bedrms": 1.0,
                "population": 800,
                "ave_occup": 3.0,
                "latitude": 34.0,
                "longitude": -118.0,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_value"] == 3.14
    assert body["model_version"] == "test"


def test_predict_rejects_invalid_input(monkeypatch):
    monkeypatch.setattr(app_module, "load_champion", lambda: None)
    app_module.model_state["model"] = DummyModel()
    app_module.model_state["version"] = "test"

    with TestClient(app_module.app) as client:
        resp = client.post("/predict", json={"med_inc": 5.0})  # missing required fields

    assert resp.status_code == 422

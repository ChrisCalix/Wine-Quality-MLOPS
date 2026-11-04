from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from fastapi.testclient import TestClient

from api.main import app, loader
from api.model_loader import LoadedModel

RECORD = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class FakeModel:
    def predict(self, frame):
        return np.array([1] * len(frame))

    def predict_proba(self, frame):
        return np.tile([0.2, 0.8], (len(frame), 1))


def setup_function():
    loader.try_load = lambda: None
    loader.loaded = LoadedModel(
        model=FakeModel(),
        model_name="wine-premium-dev-classifier",
        model_version="1",
        model_alias="champion",
        run_id="test-run",
        loaded_at=datetime.now(UTC),
    )
    loader.error = None


def test_predict_valid_record_returns_200():
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [RECORD]})
    assert response.status_code == 200
    assert response.json()["predictions"][0] == {
        "premium": 1,
        "probability_premium": 0.8,
    }


def test_predict_invalid_ph_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [{**RECORD, "ph": 20}]})
    assert response.status_code == 422


def test_ready_returns_503_when_model_not_loaded():
    loader.loaded = None
    loader.error = "simulated load failure"
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503


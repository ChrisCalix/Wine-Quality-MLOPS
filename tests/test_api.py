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


class ChampionModel:
    """Imita el modelo registrado: ``predict`` devuelve ``[P(no premium), P(premium)]``."""

    def __init__(self, p_premium: float) -> None:
        self.p_premium = p_premium

    def predict(self, frame):
        return np.tile([1 - self.p_premium, self.p_premium], (len(frame), 1))


def load_champion(p_premium: float, threshold: float) -> None:
    loader.loaded = LoadedModel(
        model=ChampionModel(p_premium),
        model_name="wine-premium-classifier",
        model_version="3",
        model_alias="champion",
        run_id="test-run",
        loaded_at=datetime.now(UTC),
        decision_threshold=threshold,
    )


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



def test_champion_probability_above_threshold_is_premium():
    load_champion(p_premium=0.30, threshold=0.22)
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [RECORD]})
    prediction = response.json()["predictions"][0]
    assert prediction["premium"] == 1
    assert prediction["probability_premium"] == 0.30


def test_champion_probability_below_threshold_is_not_premium():
    load_champion(p_premium=0.10, threshold=0.22)
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [RECORD]})
    prediction = response.json()["predictions"][0]
    assert prediction["premium"] == 0
    assert prediction["probability_premium"] == 0.10


def test_champion_threshold_changes_the_class_not_the_probability():
    load_champion(p_premium=0.30, threshold=0.50)
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [RECORD]})
    assert response.json()["predictions"][0] == {"premium": 0, "probability_premium": 0.30}


def test_champion_predict_with_several_records():
    load_champion(p_premium=0.40, threshold=0.22)
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [RECORD, RECORD, RECORD]})
    assert [p["premium"] for p in response.json()["predictions"]] == [1, 1, 1]


def test_model_info_exposes_the_decision_threshold():
    load_champion(p_premium=0.30, threshold=0.22)
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["decision_threshold"] == 0.22

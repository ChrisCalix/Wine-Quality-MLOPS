"""Pruebas del concept drift: mecanismo de alarma y comportamiento con la regla de la SPEC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from wine.config import FEATURES
from wine.drift.concept import (
    DRIFT_START,
    find_alarm,
    plot,
    relabel_fractions,
    run_concept_drift,
)

PROCESSED = Path("data/processed")
THRESHOLD = 0.22  # umbral del champion
SEEDS = range(5)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "test.csv")


@pytest.fixture(scope="module")
def predict():
    """Un Random Forest local equivalente al champion; no requiere MLflow."""
    train = pd.read_csv(PROCESSED / "train.csv")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
    model.fit(train[list(FEATURES)], train["premium"])

    def _predict(df: pd.DataFrame) -> np.ndarray:
        return (model.predict_proba(df[list(FEATURES)])[:, 1] >= THRESHOLD).astype(int)

    return _predict


def test_sin_reetiquetado_no_hay_alarma(test_df: pd.DataFrame, predict) -> None:
    for seed in SEEDS:
        result = run_concept_drift(test_df, predict, relabel=False, seed=seed)
        assert result["alarm_batch"] is None, f"alarma falsa con la semilla {seed}"


def test_el_mecanismo_alarma_cuando_el_modelo_si_se_degrada(test_df: pd.DataFrame) -> None:
    """Con un modelo que acierta las etiquetas originales, reetiquetar sí baja su F1."""

    def oracle(df: pd.DataFrame) -> np.ndarray:
        return df["premium"].to_numpy()

    for seed in SEEDS:
        result = run_concept_drift(test_df, oracle, relabel=True, seed=seed)
        assert result["alarm_batch"] is not None
        assert all(batch >= DRIFT_START for batch in result["batches_below_threshold"])


def test_con_la_regla_de_la_spec_el_champion_no_alarma(test_df: pd.DataFrame, predict) -> None:
    """Caracterización: el modelo ya usa alcohol y sulphates, así que la regla de la SPEC
    no baja su F1 y no hay alarma (ver docs/DRIFT.md §2.2)."""
    for seed in SEEDS:
        result = run_concept_drift(test_df, predict, relabel=True, seed=seed)
        assert result["alarm_batch"] is None


def test_fracciones_de_reetiquetado() -> None:
    fractions = relabel_fractions()
    assert fractions[:4] == [0.0, 0.0, 0.0, 0.0]
    assert fractions[4:11] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert fractions[11] == pytest.approx(0.7) and len(fractions) == 12


def test_la_persistencia_exige_tres_lotes_consecutivos() -> None:
    assert find_alarm([0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1], threshold=0.3) == 7
    assert find_alarm([0.5, 0.1, 0.5, 0.1, 0.5, 0.1], threshold=0.3) is None
    assert find_alarm([0.1, 0.1], threshold=0.3) is None


def test_la_grafica_se_genera(test_df: pd.DataFrame, predict, tmp_path: Path) -> None:
    result = run_concept_drift(test_df, predict)
    plot(result, tmp_path / "concept_drift.png")
    assert (tmp_path / "concept_drift.png").stat().st_size > 1000

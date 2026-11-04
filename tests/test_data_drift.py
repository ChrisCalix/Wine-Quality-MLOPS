"""Pruebas del detector de data drift: VERDE con mismo origen, ROJO con lotes derivados."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wine.config import FEATURES
from wine.drift.calibrate import calibrate
from wine.drift.data import GREEN, RED, check_batch, main, psi
from wine.drift.inject import inject

PROCESSED = Path("data/processed")


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "train.csv")


@pytest.fixture(scope="module")
def green() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "monitor_green.csv")


@pytest.fixture(scope="module")
def white() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "monitor_red_white.csv")


def test_monitor_green_es_verde(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    report = check_batch(green, baseline)
    assert report.status == GREEN
    assert report.n_variables_with_drift == 0


def test_monitor_red_white_es_rojo(baseline: pd.DataFrame, white: pd.DataFrame) -> None:
    report = check_batch(white, baseline)
    assert report.status == RED
    drifted = {v.name for v in report.variables if v.drift}
    assert {"residual_sugar", "total_sulfur_dioxide", "free_sulfur_dioxide"} <= drifted


def test_monitor_red_injected_es_rojo(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    report = check_batch(inject(green), baseline)
    assert report.status == RED
    drifted = {v.name for v in report.variables if v.drift}
    assert drifted == {"alcohol", "residual_sugar", "total_sulfur_dioxide"}


def test_el_reporte_trae_una_entrada_por_variable(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    report = check_batch(green, baseline)
    assert [v.name for v in report.variables] == list(FEATURES)


def test_chi2_detecta_cambio_en_la_clase_predicha(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    reference = (rng.random(200) < 0.13).astype(int)
    same = (rng.random(200) < 0.13).astype(int)
    shifted = (rng.random(200) < 0.45).astype(int)
    assert check_batch(green, baseline, same, reference).status == GREEN
    report = check_batch(green, baseline, shifted, reference)
    assert report.status == RED and report.chi2 is not None and report.chi2.drift


def test_chi2_se_omite_sin_predicciones(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    report = check_batch(green, baseline)
    assert report.chi2 is None and report.notes


def test_psi_identicas_y_desplazadas() -> None:
    values = np.random.default_rng(0).normal(size=2000)
    assert psi(values, values) < 0.01
    assert psi(values, values + 3) > 0.25


def test_columna_faltante_en_el_lote(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="alcohol"):
        check_batch(green.drop(columns=["alcohol"]), baseline)


def test_inyeccion_no_cambia_las_etiquetas(green: pd.DataFrame) -> None:
    injected = inject(green)
    assert (injected["quality"] == green["quality"]).all()
    assert np.allclose(injected["alcohol"], green["alcohol"] + 1.5)
    assert np.allclose(injected["residual_sugar"], green["residual_sugar"] * 3)
    assert np.allclose(injected["total_sulfur_dioxide"], green["total_sulfur_dioxide"] * 2)


def test_cli_devuelve_1_en_rojo_y_0_en_verde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, green: pd.DataFrame
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    batches = {
        "green": PROCESSED / "monitor_green.csv",
        "white": PROCESSED / "monitor_red_white.csv",
    }
    injected = tmp_path / "injected.csv"
    inject(green).to_csv(injected, index=False)
    batches["injected"] = injected

    codes = {}
    for name, batch in batches.items():
        out = tmp_path / f"{name}.json"
        codes[name] = main(["--batch", str(batch), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        assert len(report["variables"]) == len(FEATURES)
    assert codes == {"green": 0, "white": 1, "injected": 1}


def test_calibracion_sin_falsos_rojos_con_k3(baseline: pd.DataFrame, green: pd.DataFrame) -> None:
    result = calibrate(green, baseline, n_batches=30)
    k3 = next(row for row in result["rows"] if row["k"] == 3)
    assert k3["false_red_rate"] < 0.05

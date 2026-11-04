"""Pruebas del ETL: esquema, rangos, duplicados, fuga entre splits y hashes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wine.config import CSV_COLUMN_MAP, FEATURES
from wine.data.etl import run_etl
from wine.data.schema import SchemaError, validate_schema
from wine.data.split import assert_no_leakage, make_splits, save_splits

CSV_NAMES = {value: key for key, value in CSV_COLUMN_MAP.items()}


def make_wines(n: int, seed: int = 0, wine_type: str = "red") -> pd.DataFrame:
    """Vinos sintéticos dentro de los rangos de tintos, con `premium` equilibrado."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "fixed_acidity": rng.uniform(5, 12, n),
            "volatile_acidity": rng.uniform(0.2, 1.0, n),
            "citric_acid": rng.uniform(0, 0.8, n),
            "residual_sugar": rng.uniform(1, 6, n),
            "chlorides": rng.uniform(0.03, 0.2, n),
            "free_sulfur_dioxide": rng.uniform(2, 60, n),
            "total_sulfur_dioxide": rng.uniform(10, 200, n),
            "density": rng.uniform(0.990, 1.002, n),
            "ph": rng.uniform(2.9, 3.8, n),
            "sulphates": rng.uniform(0.4, 1.5, n),
            "alcohol": rng.uniform(9, 14, n),
            "quality": rng.choice([5, 6, 7, 8], n, p=[0.3, 0.3, 0.25, 0.15]),
        }
    )
    df["premium"] = (df["quality"] >= 7).astype(int)
    df["wine_type"] = wine_type
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    raw = df[[*FEATURES, "quality"]].rename(columns=CSV_NAMES)
    raw.to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def wines() -> pd.DataFrame:
    return make_wines(300)


def test_esquema_correcto(wines: pd.DataFrame) -> None:
    validate_schema(wines[[*FEATURES, "quality"]])


def test_columna_faltante_nombra_la_columna(wines: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="alcohol"):
        validate_schema(wines[[*FEATURES, "quality"]].drop(columns=["alcohol"]))


def test_columna_inesperada(wines: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="wine_type"):
        validate_schema(wines)


def test_valor_nulo(wines: pd.DataFrame) -> None:
    data = wines[[*FEATURES, "quality"]].copy()
    data.loc[0, "sulphates"] = np.nan
    with pytest.raises(SchemaError, match="sulphates"):
        validate_schema(data)


def test_tipo_no_numerico(wines: pd.DataFrame) -> None:
    data = wines[[*FEATURES, "quality"]].astype({"ph": str})
    with pytest.raises(SchemaError, match="ph"):
        validate_schema(data)


def test_valor_fuera_de_rango_de_integridad(wines: pd.DataFrame) -> None:
    data = wines[[*FEATURES, "quality"]].copy()
    data.loc[0, "ph"] = 20.0
    with pytest.raises(SchemaError, match="ph"):
        validate_schema(data, strict=False)


def test_rango_de_tintos_solo_en_modo_estricto(wines: pd.DataFrame) -> None:
    data = wines[[*FEATURES, "quality"]].copy()
    data.loc[0, "density"] = 1.039  # plausible en blancos, no en tintos
    with pytest.raises(SchemaError, match="density"):
        validate_schema(data, strict=True)
    validate_schema(data, strict=False)


def test_duplicados_eliminados_y_objetivo_derivado(
    wines: pd.DataFrame, tmp_path: Path
) -> None:
    repeated = pd.concat([wines, wines.iloc[:25]], ignore_index=True)
    result = run_etl(write_csv(repeated, tmp_path / "red.csv"), "red")
    assert len(result) == len(wines)
    assert not result.duplicated().any()
    assert (result["premium"] == (result["quality"] >= 7).astype(int)).all()
    assert set(result["wine_type"]) == {"red"}


def test_sin_fuga_entre_splits() -> None:
    splits = make_splits(make_wines(400), make_wines(600, seed=1, wine_type="white"))
    assert_no_leakage(splits)
    sizes = {name: len(df) for name, df in splits.items()}
    assert sizes["train"] == 280
    assert sizes["test"] == sizes["monitor_green"] == 60
    assert sizes["monitor_red_white"] == 500


def test_fuga_detectada() -> None:
    splits = make_splits(make_wines(400), make_wines(600, seed=1, wine_type="white"))
    splits["test"] = pd.concat([splits["test"], splits["train"].iloc[:1]])
    with pytest.raises(AssertionError, match="Fuga"):
        assert_no_leakage(splits)


def test_hashes_reproducibles(tmp_path: Path) -> None:
    red, white = make_wines(400), make_wines(600, seed=1, wine_type="white")
    first = save_splits(make_splits(red, white), tmp_path / "a")
    second = save_splits(make_splits(red, white), tmp_path / "b")
    assert first == second

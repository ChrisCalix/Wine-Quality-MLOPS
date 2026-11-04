"""Validación de esquema, tipos, nulos y rangos de los datos de vino.

Hay dos niveles de rango:

- ``INTEGRITY_RANGES``: valores físicamente imposibles. Se aplican a cualquier
  dato, incluidos los lotes de monitoreo (un lote con drift real no debe
  abortar antes de que el detector lo mida).
- ``RED_RANGES``: rangos plausibles de los tintos de entrenamiento. Solo se
  aplican al ETL de entrenamiento (``strict=True``).

Todos los números son una **[Decisión]** del equipo; los de ``RED_RANGES`` son
los de ``PLAN.md`` §3A.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from wine.config import FEATURES

TARGET_COLUMN = "quality"
REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURES, TARGET_COLUMN)

INTEGRITY_RANGES: dict[str, tuple[float, float]] = {
    "fixed_acidity": (0.0, 30.0),
    "volatile_acidity": (0.0, 5.0),
    "citric_acid": (0.0, 5.0),
    "residual_sugar": (0.0, 200.0),
    "chlorides": (0.0, 2.0),
    "free_sulfur_dioxide": (0.0, 500.0),
    "total_sulfur_dioxide": (0.0, 1000.0),
    "density": (0.9, 1.2),
    "ph": (0.0, 14.0),
    "sulphates": (0.0, 5.0),
    "alcohol": (0.0, 100.0),
    TARGET_COLUMN: (0.0, 10.0),
}

RED_RANGES: dict[str, tuple[float, float]] = {
    "ph": (2.5, 4.5),
    "alcohol": (8.0, 15.0),
    "density": (0.98, 1.01),
}


class SchemaError(ValueError):
    """Los datos no cumplen el esquema; el mensaje nombra la columna."""


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Faltan columnas: {', '.join(missing)}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        raise SchemaError(f"Columnas no esperadas: {', '.join(extra)}")


def _check_types_and_nulls(df: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        series = df[column]
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            raise SchemaError(
                f"Columna '{column}': se esperaba tipo numérico y es {series.dtype}"
            )
        nulls = int(series.isna().sum())
        if nulls:
            raise SchemaError(f"Columna '{column}': {nulls} valores nulos")


def _check_ranges(
    df: pd.DataFrame, ranges: dict[str, tuple[float, float]], label: str
) -> None:
    for column, (low, high) in ranges.items():
        series = df[column]
        outside = series[(series < low) | (series > high)]
        if not outside.empty:
            raise SchemaError(
                f"Columna '{column}': {len(outside)} valores fuera del rango "
                f"{label} [{low}, {high}] (mín={series.min()}, máx={series.max()})"
            )


def validate_schema(df: pd.DataFrame, strict: bool = True) -> None:
    """Lanza ``SchemaError`` si ``df`` no cumple el esquema.

    ``strict=True`` aplica además los rangos plausibles de los tintos; usar
    ``strict=False`` para lotes de monitoreo, donde solo importa la integridad.
    """
    _check_columns(df)
    _check_types_and_nulls(df)
    _check_ranges(df, INTEGRITY_RANGES, "de integridad")
    if strict:
        _check_ranges(df, RED_RANGES, "de tintos")

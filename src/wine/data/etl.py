"""ETL de Wine Quality: lectura, normalización, validación y limpieza.

La deduplicación ocurre aquí, antes de cualquier partición, para que filas
idénticas no caigan a la vez en ``train`` y ``test`` (D-03).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wine.config import CSV_COLUMN_MAP, FEATURES
from wine.data.schema import REQUIRED_COLUMNS, TARGET_COLUMN, validate_schema

logger = logging.getLogger(__name__)

PREMIUM_MIN_QUALITY = 7
RAW_DIR = Path("data/raw")
FILES = {
    "red": RAW_DIR / "winequality-red.csv",
    "white": RAW_DIR / "winequality-white.csv",
}


def load_raw(path: Path) -> pd.DataFrame:
    """Lee un CSV oficial (separador ``;``) y normaliza los nombres a snake_case."""
    return pd.read_csv(path, sep=";").rename(columns=CSV_COLUMN_MAP)


def drop_duplicates(df: pd.DataFrame, wine_type: str) -> pd.DataFrame:
    """Elimina filas exactamente repetidas y reporta cuántas y qué conflictos hay."""
    before = len(df)
    conflicting = int(
        df.drop_duplicates().duplicated(subset=list(FEATURES), keep=False).sum()
    )
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    logger.info(
        "%s: duplicados exactos eliminados=%d (filas %d -> %d)",
        wine_type, removed, before, len(cleaned),
    )
    if conflicting:
        logger.warning(
            "%s: %d filas comparten las 11 entradas con distinta calidad",
            wine_type, conflicting,
        )
    return cleaned


def run_etl(path: Path, wine_type: str, strict: bool = True) -> pd.DataFrame:
    """Valida y limpia un CSV; devuelve las 11 entradas, ``quality``,
    ``premium`` y ``wine_type`` (esta última es solo informativa).

    ``strict=False`` omite los rangos de tintos, para el lote de blancos.
    """
    df = load_raw(path)
    validate_schema(df, strict=strict)
    df = df[list(REQUIRED_COLUMNS)].astype({**{c: "float64" for c in FEATURES}})
    df = drop_duplicates(df, wine_type)
    df["premium"] = (df[TARGET_COLUMN] >= PREMIUM_MIN_QUALITY).astype(int)
    df["wine_type"] = wine_type
    logger.info(
        "%s: filas=%d tasa premium=%.4f", wine_type, len(df), df["premium"].mean()
    )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_etl(FILES["red"], "red", strict=True)
    run_etl(FILES["white"], "white", strict=False)


if __name__ == "__main__":
    main()

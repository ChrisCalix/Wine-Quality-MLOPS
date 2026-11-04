"""Genera ``monitor_red_injected``: ``monitor_green`` con desplazamientos sintéticos.

Los desplazamientos son los de ``docs/DRIFT.md`` §1.1 y afectan a tres
variables, el mínimo para que la regla ROJO (``k`` = 3) se cumpla.

    PYTHONPATH=src python -m wine.drift.inject
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from wine.data.schema import INTEGRITY_RANGES

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# columna -> (operación, valor)
SHIFTS: dict[str, tuple[str, float]] = {
    "alcohol": ("add", 1.5),
    "residual_sugar": ("mul", 3.0),
    "total_sulfur_dioxide": ("mul", 2.0),
}


def inject(green: pd.DataFrame) -> pd.DataFrame:
    """Aplica los desplazamientos a una copia; las etiquetas no cambian."""
    shifted = green.copy()
    for column, (operation, value) in SHIFTS.items():
        if operation == "add":
            shifted[column] = shifted[column] + value
        else:
            shifted[column] = shifted[column] * value
        low, high = INTEGRITY_RANGES[column]
        if not shifted[column].between(low, high).all():
            raise ValueError(f"'{column}' sale del rango de integridad [{low}, {high}]")
    return shifted


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--green", type=Path, default=PROCESSED_DIR / "monitor_green.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "monitor_red_injected.csv")
    args = parser.parse_args()
    injected = inject(pd.read_csv(args.green))
    injected.to_csv(args.out, index=False, lineterminator="\n")
    logger.info("%s: %d filas, desplazamientos %s", args.out, len(injected), SHIFTS)


if __name__ == "__main__":
    main()

"""Particiones estratificadas de los tintos y muestra de blancos para monitoreo.

Escribe en ``data/processed/`` los conjuntos ``train`` (70 %), ``test`` (15 %)
y ``monitor_green`` (15 %) de los tintos deduplicados, la muestra
``monitor_red_white`` de blancos, y ``checksums.json`` con su SHA-256.
``monitor_red_injected`` lo genera ``wine.drift.inject`` a partir de
``monitor_green``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from wine.config import SEED
from wine.data.download import sha256_of
from wine.data.etl import FILES, run_etl

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.70
WHITE_SAMPLE_SIZE = 500


def make_splits(red: pd.DataFrame, white: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, rest = train_test_split(
        red,
        train_size=TRAIN_FRACTION,
        stratify=red["premium"],
        random_state=SEED,
    )
    test, monitor_green = train_test_split(
        rest, test_size=0.5, stratify=rest["premium"], random_state=SEED
    )
    monitor_red_white = white.sample(n=WHITE_SAMPLE_SIZE, random_state=SEED)
    splits = {
        "train": train,
        "test": test,
        "monitor_green": monitor_green,
        "monitor_red_white": monitor_red_white,
    }
    return {name: df.reset_index(drop=True) for name, df in splits.items()}


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Los conjuntos de tintos no comparten ninguna fila."""
    names = ["train", "test", "monitor_green"]
    keys = {n: set(map(tuple, splits[n].drop(columns="wine_type").to_numpy())) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = keys[a] & keys[b]
            if shared:
                raise AssertionError(f"Fuga: {len(shared)} filas en {a} y {b}")


def save_splits(
    splits: dict[str, pd.DataFrame], directory: Path = PROCESSED_DIR
) -> dict[str, dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, dict[str, object]] = {}
    for name, df in splits.items():
        path = directory / f"{name}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        checksums[name] = {
            "rows": len(df),
            "premium_rate": round(float(df["premium"].mean()), 4),
            "sha256": sha256_of(path),
        }
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checksums


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    red = run_etl(FILES["red"], "red", strict=True)
    white = run_etl(FILES["white"], "white", strict=False)
    splits = make_splits(red, white)
    assert_no_leakage(splits)
    for name, info in save_splits(splits).items():
        logger.info(
            "%s: filas=%s tasa premium=%s sha256=%s",
            name, info["rows"], info["premium_rate"], str(info["sha256"])[:12],
        )


if __name__ == "__main__":
    main()

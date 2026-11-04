"""Genera ``ui/js/trainingStats.js`` con los rangos de entrenamiento y los presets de la UI.

Los rangos salen de los tintos de ``train``; el preset "tinto típico" es su mediana y el
"blanco típico" es la mediana de la muestra de blancos. Regenerar si cambian los splits:

    PYTHONPATH=src python scripts/gen_ui_stats.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wine.config import FEATURES

PROCESSED = Path("data/processed")
OUT = Path("ui/js/trainingStats.js")


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def main() -> None:
    red = pd.read_csv(PROCESSED / "train.csv")[list(FEATURES)]
    white = pd.read_csv(PROCESSED / "monitor_red_white.csv")[list(FEATURES)]
    stats = red.quantile([0, 0.05, 0.95, 1]).T
    stats.columns = ["min", "p5", "p95", "max"]

    lines = [
        "// GENERADO por scripts/gen_ui_stats.py; no editar a mano.",
        f"// Rangos: tintos de data/processed/train.csv ({len(red)} filas).",
        f"// Presets: mediana de los tintos de train y de la muestra de blancos ({len(white)} filas).",
        "",
        "export const TRAINING_STATS = {",
    ]
    for name in FEATURES:
        row = stats.loc[name]
        lines.append(
            f"  {name}: {{ min: {fmt(row['min'])}, p5: {fmt(row['p5'])}, "
            f"p95: {fmt(row['p95'])}, max: {fmt(row['max'])} }},"
        )
    lines += ["};", "", "export const PRESET_VALUES = {"]
    for key, frame in (("red", red), ("white", white)):
        values = ", ".join(f"{name}: {fmt(frame[name].median())}" for name in FEATURES)
        lines.append(f"  {key}: {{ {values} }},")
    lines += ["};", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"escrito {OUT}")


if __name__ == "__main__":
    main()

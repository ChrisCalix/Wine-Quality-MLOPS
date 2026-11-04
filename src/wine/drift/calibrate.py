"""Calibra la regla "k de 11" del data drift con lotes bootstrap de ``monitor_green``.

Criterio (``SPEC.md`` §7.1): menos de 5 % de falsos ROJOS en 200 lotes bootstrap
de un lote del mismo origen. Para cada ``k`` también se reporta si los lotes
``monitor_red_white`` y ``monitor_red_injected`` siguen siendo ROJO.

    PYTHONPATH=src python -m wine.drift.calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wine.config import MODEL_URI, SEED
from wine.drift.data import check_batch

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
N_BATCHES = 200
MAX_FALSE_RED = 0.05
K_VALUES = range(1, 7)


def calibrate(
    green: pd.DataFrame,
    baseline: pd.DataFrame,
    n_batches: int = N_BATCHES,
    seed: int = SEED,
    green_pred: np.ndarray | None = None,
    reference_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Lotes bootstrap de ``green`` (con reemplazo, mismo tamaño) y tasa de falsos ROJOS por k."""
    rng = np.random.default_rng(seed)
    drift_counts, chi2_flags = [], []
    for _ in range(n_batches):
        idx = rng.integers(0, len(green), len(green))
        batch_pred = None if green_pred is None else green_pred[idx]
        report = check_batch(green.iloc[idx], baseline, batch_pred, reference_pred, min_drift_variables=99)
        drift_counts.append(report.n_variables_with_drift)
        chi2_flags.append(bool(report.chi2 and report.chi2.drift))
    counts, flags = np.array(drift_counts), np.array(chi2_flags)
    rows = [
        {
            "k": k,
            "false_red_rate": float(np.mean((counts >= k) | flags)),
            "false_red_batches": int(np.sum((counts >= k) | flags)),
        }
        for k in K_VALUES
    ]
    return {
        "n_batches": n_batches,
        "seed": seed,
        "with_chi2": green_pred is not None,
        "chi2_false_red_batches": int(flags.sum()),
        "max_variables_with_drift": int(counts.max()),
        "rows": rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports/calibration_data_drift.json"))
    parser.add_argument("--model-uri", help="models:/<nombre>@<alias>")
    args = parser.parse_args()

    green = pd.read_csv(PROCESSED_DIR / "monitor_green.csv")
    baseline = pd.read_csv(PROCESSED_DIR / "train.csv")
    white = pd.read_csv(PROCESSED_DIR / "monitor_red_white.csv")
    injected = pd.read_csv(PROCESSED_DIR / "monitor_red_injected.csv")
    model_uri = args.model_uri or (MODEL_URI if os.environ.get("MLFLOW_TRACKING_URI") else None)

    green_pred = reference_pred = None
    if model_uri:
        from wine.drift.predictor import load_predictor

        predict, _ = load_predictor(model_uri)
        green_pred, reference_pred = predict(green), predict(baseline)

    result = calibrate(green, baseline, green_pred=green_pred, reference_pred=reference_pred)
    for key, batch in (("monitor_red_white", white), ("monitor_red_injected", injected)):
        n_drift = check_batch(batch, baseline, min_drift_variables=99).n_variables_with_drift
        result[key + "_variables_with_drift"] = n_drift
    for row in result["rows"]:
        row["detects_white"] = result["monitor_red_white_variables_with_drift"] >= row["k"]
        row["detects_injected"] = result["monitor_red_injected_variables_with_drift"] >= row["k"]
        row["meets_criterion"] = row["false_red_rate"] < MAX_FALSE_RED

    logger.info("chi2 %s", "incluido" if result["with_chi2"] else "omitido (sin modelo)")
    logger.info("| k | falsos ROJOS (de %d) | tasa | detecta blancos | detecta inyectado |", result["n_batches"])
    for row in result["rows"]:
        logger.info(
            "| %d | %d | %.1f %% | %s | %s |", row["k"], row["false_red_batches"],
            100 * row["false_red_rate"], row["detects_white"], row["detects_injected"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

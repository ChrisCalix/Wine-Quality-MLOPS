"""Detector de data drift por lote: KS, PSI por variable y chi² de la clase predicha.

Diseño y umbrales: ``docs/DRIFT.md`` §1 y ``SPEC.md`` §7.1.

    PYTHONPATH=src python -m wine.drift.data --batch data/processed/monitor_green.csv \
        --out reports/data_drift_monitor_green.json

Termina con código 1 si el lote es ROJO, para que CI pueda bloquear una promoción.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from wine.config import (
    CHI2_P_VALUE,
    DRIFT_VARIABLES_FOR_RED,
    FEATURES,
    KS_BONFERRONI_P_VALUE,
    MODEL_URI,
    PSI_STABLE_MAX,
    PSI_WATCH_MAX,
)

logger = logging.getLogger(__name__)

GREEN, YELLOW, RED = "VERDE", "AMARILLO", "ROJO"
PSI_BINS = 10
PSI_EPSILON = 1e-4
PROCESSED_DIR = Path("data/processed")


@dataclass
class VariableResult:
    name: str
    ks_statistic: float
    ks_p_value: float
    psi: float
    ks_drift: bool
    psi_drift: bool
    drift: bool  # KS significativo y PSI >= PSI_WATCH_MAX


@dataclass
class Chi2Result:
    statistic: float
    p_value: float
    batch_premium_rate: float
    reference_premium_rate: float
    drift: bool


@dataclass
class DriftReport:
    status: str
    n_variables_with_drift: int
    min_drift_variables: int
    n_batch: int
    n_baseline: int
    variables: list[VariableResult] = field(default_factory=list)
    chi2: Chi2Result | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def psi(baseline: np.ndarray, batch: np.ndarray, bins: int = PSI_BINS) -> float:
    """PSI con intervalos por cuantiles del baseline y extremos abiertos.

    Los valores fuera del rango del baseline caen en el primer o último
    intervalo; los intervalos vacíos se suavizan con un epsilon.
    """
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(baseline, bins=edges)[0] / len(baseline)
    actual = np.histogram(batch, bins=edges)[0] / len(batch)
    expected = np.clip(expected, PSI_EPSILON, None)
    actual = np.clip(actual, PSI_EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _check_features(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas {', '.join(missing)}")
    if df[list(FEATURES)].isna().any().any():
        raise ValueError(f"{label}: hay valores nulos en las variables de entrada")


def _chi2(batch_pred: np.ndarray, reference_pred: np.ndarray) -> Chi2Result:
    """Compara la proporción de ``premium`` predicha entre el lote y la referencia."""
    table = np.array(
        [
            [int(batch_pred.sum()), int(len(batch_pred) - batch_pred.sum())],
            [int(reference_pred.sum()), int(len(reference_pred) - reference_pred.sum())],
        ]
    )
    statistic, p_value, _, _ = stats.chi2_contingency(table)
    return Chi2Result(
        statistic=float(statistic),
        p_value=float(p_value),
        batch_premium_rate=float(batch_pred.mean()),
        reference_premium_rate=float(reference_pred.mean()),
        drift=bool(p_value < CHI2_P_VALUE),
    )


def check_batch(
    batch_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    batch_pred: np.ndarray | None = None,
    reference_pred: np.ndarray | None = None,
    min_drift_variables: int = DRIFT_VARIABLES_FOR_RED,
) -> DriftReport:
    """Compara un lote contra el baseline y devuelve el estado VERDE/AMARILLO/ROJO.

    ``batch_pred`` y ``reference_pred`` son las clases premium predichas por el
    modelo para el lote y para una referencia no vista en el entrenamiento; con
    ambas se calcula el chi². Sin ellas, el chi² se omite y se anota en el reporte.
    """
    _check_features(batch_df, "lote")
    _check_features(baseline_df, "baseline")

    variables = []
    for name in FEATURES:
        base = baseline_df[name].to_numpy(dtype=float)
        new = batch_df[name].to_numpy(dtype=float)
        ks = stats.ks_2samp(base, new)
        value = psi(base, new)
        ks_drift = bool(ks.pvalue < KS_BONFERRONI_P_VALUE)
        psi_drift = value >= PSI_WATCH_MAX
        variables.append(
            VariableResult(
                name=name,
                ks_statistic=float(ks.statistic),
                ks_p_value=float(ks.pvalue),
                psi=value,
                ks_drift=ks_drift,
                psi_drift=psi_drift,
                drift=ks_drift and psi_drift,
            )
        )

    notes: list[str] = []
    chi2 = None
    if batch_pred is not None and reference_pred is not None:
        chi2 = _chi2(np.asarray(batch_pred), np.asarray(reference_pred))
    else:
        notes.append("chi2 omitido: no se proporcionaron predicciones del modelo")

    n_drift = sum(v.drift for v in variables)
    watch = any(v.psi >= PSI_STABLE_MAX for v in variables)
    if n_drift >= min_drift_variables or (chi2 is not None and chi2.drift):
        status = RED
    elif n_drift >= 1 or watch:
        status = YELLOW
    else:
        status = GREEN
    return DriftReport(
        status=status,
        n_variables_with_drift=n_drift,
        min_drift_variables=min_drift_variables,
        n_batch=len(batch_df),
        n_baseline=len(baseline_df),
        variables=variables,
        chi2=chi2,
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True, help="CSV del lote")
    parser.add_argument("--baseline", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument(
        "--reference", type=Path, default=PROCESSED_DIR / "test.csv",
        help="CSV no visto en el entrenamiento, para la distribución de clase predicha",
    )
    parser.add_argument("--out", type=Path, help="ruta del reporte JSON")
    parser.add_argument(
        "--model-uri",
        help="models:/<nombre>@<alias>; por defecto el champion si MLFLOW_TRACKING_URI está definido",
    )
    args = parser.parse_args(argv)

    batch = pd.read_csv(args.batch)
    baseline = pd.read_csv(args.baseline)
    model_uri = args.model_uri or (MODEL_URI if os.environ.get("MLFLOW_TRACKING_URI") else None)

    batch_pred = reference_pred = None
    if model_uri:
        from wine.drift.predictor import load_predictor

        predict, threshold = load_predictor(model_uri)
        reference = pd.read_csv(args.reference)
        batch_pred, reference_pred = predict(batch), predict(reference)
        logger.info("modelo %s (umbral %.2f)", model_uri, threshold)

    report = check_batch(batch, baseline, batch_pred, reference_pred)
    for var in report.variables:
        flag = "DRIFT" if var.drift else "ok"
        logger.info("%-22s ks_p=%.2e psi=%.3f %s", var.name, var.ks_p_value, var.psi, flag)
    if report.chi2:
        logger.info("chi2 clase predicha: p=%.3g drift=%s", report.chi2.p_value, report.chi2.drift)
    for note in report.notes:
        logger.warning(note)
    logger.info("Estado del lote: %s (%d variables con drift)", report.status, report.n_variables_with_drift)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 1 if report.status == RED else 0


if __name__ == "__main__":
    sys.exit(main())

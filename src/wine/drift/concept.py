"""Simulación de concept drift por reetiquetado y alarma de reentrenamiento.

Diseño y umbrales: ``docs/DRIFT.md`` §2 y ``SPEC.md`` §7.2. Las entradas de los
lotes no cambian; desde el lote 5 una fracción creciente de filas se reetiqueta
con la regla del nuevo panel de catadores. El F1 de la clase premium se mide por
lote con el modelo ``champion``; la alarma exige 3 lotes consecutivos bajo
``F1_ref - 3 * sigma_ref``.

    PYTHONPATH=src MLFLOW_TRACKING_URI=http://host.docker.internal:5678 \
        python -m wine.drift.concept
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from wine.config import (
    CONCEPT_DRIFT_PERSISTENCE,
    CONCEPT_DRIFT_SIGMA,
    FEATURES,
    MODEL_URI,
    SEED,
)
from wine.drift.predictor import Predictor

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
TARGET = "premium"
N_BATCHES = 12
BATCH_SIZE = 200
DRIFT_START = 5
FRACTION_STEP = 0.10
MAX_FRACTION = 0.70
REFERENCE_BATCHES = 200
NEW_RULE_ALCOHOL = 11.5
NEW_RULE_SULPHATES = 0.65


def new_panel_label(df: pd.DataFrame) -> pd.Series:
    """Regla del nuevo panel (SPEC §7.2): premium solo con ``alcohol >= 11.5`` y ``sulphates >= 0.65``."""
    return ((df["alcohol"] >= NEW_RULE_ALCOHOL) & (df["sulphates"] >= NEW_RULE_SULPHATES)).astype(int)


def relabel_fractions(
    n_batches: int = N_BATCHES, start: int = DRIFT_START
) -> list[float]:
    """0 antes del lote ``start``; luego 10 %, 20 %, ... con tope de 70 %."""
    return [
        0.0 if batch < start else min(FRACTION_STEP * (batch - start + 1), MAX_FRACTION)
        for batch in range(1, n_batches + 1)
    ]


def f1_premium(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def reference_f1(
    test: pd.DataFrame, predict: Predictor, rng: np.random.Generator,
    n_batches: int = REFERENCE_BATCHES, batch_size: int = BATCH_SIZE,
) -> tuple[float, float]:
    """Media y desviación del F1 en lotes bootstrap de ``test`` sin reetiquetar."""
    pred = predict(test)
    y = test[TARGET].to_numpy()
    scores = []
    for _ in range(n_batches):
        idx = rng.integers(0, len(test), batch_size)
        scores.append(f1_premium(y[idx], pred[idx]))
    return float(np.mean(scores)), float(np.std(scores))


def simulate(
    test: pd.DataFrame, predict: Predictor, fractions: list[float],
    rng: np.random.Generator, batch_size: int = BATCH_SIZE,
) -> list[float]:
    """F1 premium de cada lote bootstrap con la fracción de reetiquetado indicada."""
    pred = predict(test)
    labels = test[TARGET].to_numpy()
    new_labels = new_panel_label(test).to_numpy()
    scores = []
    for fraction in fractions:
        idx = rng.integers(0, len(test), batch_size)
        y = labels[idx].copy()
        chosen = rng.choice(batch_size, size=round(fraction * batch_size), replace=False)
        y[chosen] = new_labels[idx][chosen]
        scores.append(f1_premium(y, pred[idx]))
    return scores


def find_alarm(
    f1_scores: list[float], threshold: float, persistence: int = CONCEPT_DRIFT_PERSISTENCE
) -> int | None:
    """Lote (base 1) en que se completan ``persistence`` lotes consecutivos bajo el umbral."""
    run = 0
    for number, score in enumerate(f1_scores, start=1):
        run = run + 1 if score < threshold else 0
        if run >= persistence:
            return number
    return None


def run_concept_drift(
    test: pd.DataFrame, predict: Predictor, relabel: bool = True, seed: int = SEED
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    f1_ref, sigma_ref = reference_f1(test, predict, rng)
    threshold = f1_ref - CONCEPT_DRIFT_SIGMA * sigma_ref
    fractions = relabel_fractions() if relabel else [0.0] * N_BATCHES
    f1_scores = simulate(test, predict, fractions, rng)
    return {
        "seed": seed,
        "relabel": relabel,
        "n_batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "drift_start_batch": DRIFT_START if relabel else None,
        "relabel_fractions": fractions,
        "f1_ref": f1_ref,
        "sigma_ref": sigma_ref,
        "threshold": threshold,
        "persistence": CONCEPT_DRIFT_PERSISTENCE,
        "f1_per_batch": f1_scores,
        "batches_below_threshold": [i for i, s in enumerate(f1_scores, 1) if s < threshold],
        "alarm_batch": find_alarm(f1_scores, threshold),
    }


def plot(result: dict[str, Any], path: Path) -> None:
    scores = result["f1_per_batch"]
    batches = list(range(1, len(scores) + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(batches, scores, marker="o", color="#1f77b4", label="F1 premium por lote")
    ax.axhline(result["f1_ref"], color="gray", linestyle=":", label="F1_ref")
    ax.axhline(result["threshold"], color="#d62728", linestyle="--", label="Umbral F1_ref − 3σ")
    if result["drift_start_batch"]:
        ax.axvline(result["drift_start_batch"], color="#ff7f0e", linestyle="-.", label="Inicio del drift")
    if result["alarm_batch"]:
        alarm = result["alarm_batch"]
        ax.scatter([alarm], [scores[alarm - 1]], s=160, color="#d62728", zorder=5, label=f"Alarma (lote {alarm})")
    ax.set_xlabel("Lote")
    ax.set_ylabel("F1 de la clase premium")
    ax.set_xticks(batches)
    ax.set_title("Concept drift por reetiquetado progresivo")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    from wine.drift.predictor import load_predictor

    predict, model_threshold = load_predictor(args.model_uri)
    test = pd.read_csv(args.test)
    missing = [c for c in (*FEATURES, TARGET) if c not in test.columns]
    if missing:
        raise ValueError(f"{args.test}: faltan columnas {', '.join(missing)}")

    result = run_concept_drift(test, predict)
    result.update(model_uri=args.model_uri, model_decision_threshold=model_threshold)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "concept_drift.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(result, args.out_dir / "concept_drift.png")

    logger.info("F1_ref=%.3f sigma_ref=%.3f umbral=%.3f", result["f1_ref"], result["sigma_ref"], result["threshold"])
    logger.info("F1 por lote: %s", [round(s, 2) for s in result["f1_per_batch"]])
    logger.info("lotes bajo el umbral: %s | alarma en el lote: %s", result["batches_below_threshold"], result["alarm_batch"])


if __name__ == "__main__":
    main()

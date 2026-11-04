"""Entrenamiento reproducible de un candidato con tracking completo en MLflow.

Un run = una familia de modelo + hiperparámetros. La familia, los
hiperparámetros y el umbral se eligen con la validación cruzada de ``train``;
``test`` se evalúa una sola vez (``docs/crispdm.md`` §4).

    PYTHONPATH=src python -m wine.train.train --family logistic_regression \
        --params '{"C": 1.0, "class_weight": "balanced"}' --purpose baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wine.config import EXPERIMENT_NAME, FEATURES, SEED
from wine.data.download import sha256_of
from wine.data.etl import FILES
from wine.train.metrics import compute_metrics, make_figures, report_dict

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
TARGET = "premium"
CV_FOLDS = 5
PURPOSES = ("baseline", "candidate", "retrain")

FAMILIES: dict[str, tuple[type, bool]] = {
    # familia -> (clase de scikit-learn, ¿lleva StandardScaler?)
    "logistic_regression": (LogisticRegression, True),
    "random_forest": (RandomForestClassifier, False),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, False),
    "svc": (SVC, True),
}
FIXED_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"max_iter": 1000, "random_state": SEED},
    "random_forest": {"random_state": SEED},
    "hist_gradient_boosting": {"random_state": SEED},
    "svc": {"kernel": "rbf", "probability": True, "random_state": SEED},
}


def build_pipeline(family: str, params: dict[str, Any]) -> Pipeline:
    if family not in FAMILIES:
        raise ValueError(f"Familia desconocida: {family}. Opciones: {sorted(FAMILIES)}")
    estimator_cls, scaled = FAMILIES[family]
    estimator = estimator_cls(**{**FIXED_PARAMS[family], **params})
    steps: list[tuple[str, Any]] = []
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def load_split(name: str, directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_csv(directory / f"{name}.csv")


def cross_validate(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, threshold: float
) -> tuple[float, float]:
    """F1 de la clase premium por fold (CV estratificada); devuelve media y desviación."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in folds.split(x, y):
        model = clone(pipeline).fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(x.iloc[val_idx])[:, 1]
        scores.append(compute_metrics(y.iloc[val_idx].to_numpy(), proba, threshold)["f1_premium"])
    return float(np.mean(scores)), float(np.std(scores))


def baseline_stats(train: pd.DataFrame) -> dict[str, Any]:
    """Estadísticos del ``train`` que usa el detector de drift como baseline."""
    deciles = [i / 10 for i in range(11)]
    return {
        "n_rows": len(train),
        "premium_rate": float(train[TARGET].mean()),
        "features": {
            c: {
                "mean": float(train[c].mean()),
                "std": float(train[c].std()),
                "min": float(train[c].min()),
                "max": float(train[c].max()),
                "deciles": [float(v) for v in train[c].quantile(deciles)],
            }
            for c in FEATURES
        },
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dataset(df: pd.DataFrame, name: str) -> mlflow.data.Dataset:
    return mlflow.data.from_pandas(
        df[[*FEATURES, TARGET]],
        source=str(PROCESSED_DIR / f"{name}.csv"),
        name=name,
        targets=TARGET,
    )


def train_run(
    family: str,
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
    purpose: str = "candidate",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Entrena, evalúa y registra un run completo. Devuelve sus métricas."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose debe ser uno de {PURPOSES}")
    params = params or {}
    train, test = load_split("train"), load_split("test")
    x_train, y_train = train[list(FEATURES)], train[TARGET]
    x_test, y_test = test[list(FEATURES)], test[TARGET]
    pipeline = build_pipeline(family, params)

    mlflow.set_experiment(EXPERIMENT_NAME)
    name = run_name or f"{family}-{purpose}"
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "model_family": family,
                **params,
                "seed": SEED,
                "train_size": len(train),
                "test_size": len(test),
                "cv_folds": CV_FOLDS,
                "decision_threshold": threshold,
                "class_weight": params.get("class_weight"),
                "n_features": len(FEATURES),
            }
        )
        mlflow.set_tags(
            {
                "project": "wine-mlops",
                "owner": os.environ.get("WINE_OWNER") or _git("config", "user.name"),
                "git_commit": _git("rev-parse", "HEAD"),
                "dataset_version": "uci-186",
                "dataset_sha256_red": sha256_of(FILES["red"]),
                "dataset_sha256_white": sha256_of(FILES["white"]),
                "purpose": purpose,
                "crispdm_phase": "modeling",
            }
        )
        mlflow.log_input(_dataset(train, "train"), context="training")
        mlflow.log_input(_dataset(test, "test"), context="testing")

        cv_mean, cv_std = cross_validate(pipeline, x_train, y_train, threshold)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_time = time.perf_counter() - started

        proba = pipeline.predict_proba(x_test)[:, 1]
        scores = compute_metrics(y_test.to_numpy(), proba, threshold)
        metrics = {
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
            **{f"test_{key}": value for key, value in scores.items()},
            "train_time_s": train_time,
        }
        mlflow.log_metrics(metrics)

        for filename, figure in make_figures(y_test.to_numpy(), proba, threshold).items():
            mlflow.log_figure(figure, filename)
            plt.close(figure)
        mlflow.log_dict(report_dict(y_test.to_numpy(), proba, threshold), "classification_report.json")
        mlflow.log_dict(baseline_stats(train), "baseline_stats.json")
        importance = permutation_importance(
            pipeline, x_test, y_test, scoring="f1", n_repeats=5, random_state=SEED
        )
        mlflow.log_dict(
            dict(zip(FEATURES, (float(v) for v in importance.importances_mean), strict=True)),
            "feature_importance.json",
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(x_train, pipeline.predict(x_train)),
            input_example=x_train.head(5),
        )
    logger.info("%s: test_f1_premium=%.4f cv_f1_mean=%.4f", name, scores["f1_premium"], cv_mean)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--params", default="{}", help="hiperparámetros en JSON")
    parser.add_argument("--run-name")
    parser.add_argument("--purpose", default="candidate", choices=PURPOSES)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    train_run(args.family, json.loads(args.params), args.run_name, args.purpose, args.threshold)


if __name__ == "__main__":
    main()

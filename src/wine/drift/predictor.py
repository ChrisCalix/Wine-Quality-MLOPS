"""Carga el modelo registrado y devuelve una función que predice la clase premium.

El modelo registrado devuelve ``[P(no premium), P(premium)]``; el umbral de
decisión está en el tag ``decision_threshold`` de la versión del registro.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from wine.config import FEATURES

Predictor = Callable[[pd.DataFrame], np.ndarray]


def parse_alias_uri(model_uri: str) -> tuple[str, str]:
    """``models:/<nombre>@<alias>`` -> (nombre, alias)."""
    prefix = "models:/"
    if not model_uri.startswith(prefix) or "@" not in model_uri:
        raise ValueError(f"Se esperaba models:/<nombre>@<alias>, no {model_uri!r}")
    name, alias = model_uri[len(prefix):].split("@", 1)
    return name, alias


def load_predictor(model_uri: str) -> tuple[Predictor, float]:
    """Devuelve (predictor de clases 0/1, umbral usado)."""
    import mlflow
    from mlflow import MlflowClient

    name, alias = parse_alias_uri(model_uri)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    threshold = float(version.tags.get("decision_threshold", 0.5))
    model = mlflow.pyfunc.load_model(model_uri)

    def predict(df: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(model.predict(df[list(FEATURES)]))[:, 1]
        return (proba >= threshold).astype(int)

    return predict, threshold

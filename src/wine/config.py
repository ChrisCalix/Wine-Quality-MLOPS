"""Contratos compartidos del proyecto Wine Quality MLOps.

Este módulo solo declara constantes para que pueda importarse desde scripts,
la API y las pruebas sin descargar datos, iniciar servicios ni modificar el
entorno.
"""

from __future__ import annotations

from typing import Final

SEED: Final[int] = 42

FEATURES: Final[tuple[str, ...]] = (
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
)

CSV_COLUMN_MAP: Final[dict[str, str]] = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "ph",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}

EXPERIMENT_NAME: Final[str] = "wine-premium"
DEV_EXPERIMENT_NAME: Final[str] = "wine-premium-dev"
REGISTERED_MODEL_NAME: Final[str] = "wine-premium-classifier"
DEV_REGISTERED_MODEL_NAME: Final[str] = "wine-premium-dev-classifier"
CHAMPION_ALIAS: Final[str] = "champion"
CHALLENGER_ALIAS: Final[str] = "challenger"
MODEL_URI: Final[str] = (
    f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
)

# Umbrales definidos en SPEC.md §7.1.
KS_ALPHA: Final[float] = 0.05
KS_BONFERRONI_P_VALUE: Final[float] = KS_ALPHA / len(FEATURES)
PSI_STABLE_MAX: Final[float] = 0.10
PSI_WATCH_MAX: Final[float] = 0.25
CHI2_P_VALUE: Final[float] = 0.01
DRIFT_VARIABLES_FOR_RED: Final[int] = 3

# Umbrales definidos en SPEC.md §7.2.
CONCEPT_DRIFT_SIGMA: Final[float] = 3.0
CONCEPT_DRIFT_PERSISTENCE: Final[int] = 3


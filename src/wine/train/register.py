"""Registra v1, v2 y v3 de ``wine-premium-classifier`` y asigna los aliases.

Selección (``docs/crispdm.md`` §5), siempre por ``cv_f1_mean`` y nunca por
``test``:

- v1: run baseline (regresión logística, umbral 0,5).
- v3: mejor run del barrido con umbral ajustado por CV  -> alias ``champion``.
- v2: mejor run de otra familia que v3                   -> alias ``challenger``.

    PYTHONPATH=src MLFLOW_TRACKING_URI=http://localhost:5678 \
        python -m wine.train.register
"""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

from wine.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MODEL_URI,
    REGISTERED_MODEL_NAME,
)

logger = logging.getLogger(__name__)

BASELINE_RUN_NAME = "01-lr-C1.0-balanced"
TUNED_SWEEP = "sweep.yaml+tuned"
BASELINE_SWEEP = "sweep.yaml"

# Criterios de aceptación de docs/crispdm.md §5 (medidos sobre test).
MIN_RECALL = 0.70
MIN_PRECISION = 0.50
MIN_F1 = 0.60


def find_runs(client: MlflowClient, sweep: str) -> list[Run]:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento {EXPERIMENT_NAME}")
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.sweep = '{sweep}'"
    )


def select_runs(client: MlflowClient) -> tuple[Run, Run, Run]:
    """Devuelve (v1, v2, v3) según la regla de selección."""
    baseline = [
        r for r in find_runs(client, BASELINE_SWEEP) if r.info.run_name == BASELINE_RUN_NAME
    ]
    tuned = sorted(
        find_runs(client, TUNED_SWEEP),
        key=lambda r: r.data.metrics["cv_f1_mean"],
        reverse=True,
    )
    if not baseline or not tuned:
        raise RuntimeError("Faltan runs: ejecuta los barridos antes de registrar")
    best = tuned[0]
    family = best.data.params["model_family"]
    others = [r for r in tuned if r.data.params["model_family"] != family]
    return baseline[0], others[0], best


def logged_model_id(client: MlflowClient, run: Run) -> str:
    """Id ``m-…`` del modelo que registró este run (MLflow 3)."""
    found = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
    )
    if not found:
        raise RuntimeError(f"El run {run.info.run_name} no tiene modelo registrado")
    return found[0].model_id


def meets_criteria(run: Run, baseline: Run) -> bool:
    m = run.data.metrics
    return (
        m["test_recall_premium"] >= MIN_RECALL
        and m["test_precision_premium"] >= MIN_PRECISION
        and m["test_f1_premium"] >= MIN_F1
        and m["test_f1_premium"] > baseline.data.metrics["test_f1_premium"]
    )


def describe(run: Run, role: str) -> str:
    p, m = run.data.params, run.data.metrics
    return (
        f"{role}. Run '{run.info.run_name}' ({p['model_family']}), umbral "
        f"{p['decision_threshold']} ({p['threshold_source']}). "
        f"cv_f1_mean={m['cv_f1_mean']:.3f}, test_f1_premium={m['test_f1_premium']:.3f}, "
        f"precisión={m['test_precision_premium']:.3f}, recall={m['test_recall_premium']:.3f}."
    )


def register(client: MlflowClient) -> dict[str, str]:
    """Registra las tres versiones y asigna los aliases; devuelve {nombre: versión}."""
    try:
        existing = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        existing = []
    if existing:
        raise RuntimeError(
            f"{REGISTERED_MODEL_NAME} ya tiene {len(existing)} versiones; "
            "no se vuelve a registrar para no duplicarlas"
        )

    v1, v2, v3 = select_runs(client)
    client.create_registered_model(
        REGISTERED_MODEL_NAME,
        description=(
            "Clasificador de vinos tintos premium (quality >= 7), UCI Wine Quality 186. "
            "Devuelve [P(no premium), P(premium)]; el umbral de decisión está en el "
            "tag decision_threshold de cada versión."
        ),
    )
    plan = [
        ("v1", v1, "Línea base: regresión logística con umbral 0,5"),
        ("v2", v2, "Candidato: mejor run de otra familia que el champion"),
        ("v3", v3, "Mejor run por cv_f1_mean con umbral ajustado por CV"),
    ]
    versions: dict[str, str] = {}
    for label, run, role in plan:
        model_id = logged_model_id(client, run)
        created = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=f"models:/{model_id}",
            model_id=model_id,
            run_id=run.info.run_id,
            description=describe(run, role),
            tags={
                "validated": str(meets_criteria(run, v1)).lower(),
                "source_run": run.info.run_id,
                "test_f1_premium": f"{run.data.metrics['test_f1_premium']:.4f}",
                "decision_threshold": run.data.params["decision_threshold"],
            },
        )
        versions[label] = created.version
        logger.info("%s -> versión %s (%s)", label, created.version, run.info.run_name)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, versions["v3"])
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, versions["v2"])
    logger.info("champion -> v%s, challenger -> v%s", versions["v3"], versions["v2"])
    return versions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    register(MlflowClient())
    logger.info("Referencia del servicio: %s", MODEL_URI)


if __name__ == "__main__":
    main()

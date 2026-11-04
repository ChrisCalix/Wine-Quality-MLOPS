"""Ejecuta el barrido de ``experiments/sweep.yaml``: un run de MLflow por combinación.

    PYTHONPATH=src python -m wine.train.sweep --config experiments/sweep.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from wine.train.train import train_run

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not config.get("runs"):
        raise ValueError(f"{path} no define ningún run")
    return config


def run_sweep(path: Path, tune: bool = False) -> dict[str, dict[str, float]]:
    config = load_config(path)
    threshold = float(config.get("threshold", 0.5))
    results: dict[str, dict[str, float]] = {}
    suffix = "-thr" if tune else ""
    for entry in config["runs"]:
        results[entry["name"] + suffix] = train_run(
            family=entry["family"],
            params=entry.get("params") or {},
            run_name=entry["name"] + suffix,
            purpose=entry.get("purpose", "candidate"),
            threshold=threshold,
            tune=tune,
            extra_tags={"sweep": path.name + ("+tuned" if tune else "")},
        )
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("experiments/sweep.yaml"))
    parser.add_argument(
        "--tune-threshold", action="store_true",
        help="cada run elige su umbral con predicciones fuera de fold",
    )
    args = parser.parse_args()
    results = run_sweep(args.config, args.tune_threshold)
    # La elección usa solo la validación cruzada de train (crispdm.md §4).
    best = max(results, key=lambda name: results[name]["cv_f1_mean"])
    logger.info("Mejor por cv_f1_mean: %s", best)


if __name__ == "__main__":
    main()

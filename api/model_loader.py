"""Carga el modelo servido desde MLflow Model Registry."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient


@dataclass
class LoadedModel:
    model: object
    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    loaded_at: datetime


class ModelLoader:
    """Mantiene el modelo cargado y sus metadatos de Registry."""

    def __init__(self) -> None:
        self.model_uri = os.getenv(
            "MODEL_URI", "models:/wine-premium-dev-classifier@champion"
        )
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5678")
        self.loaded: LoadedModel | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.loaded is not None

    def load(self) -> LoadedModel:
        """Carga el alias configurado y obtiene versión/run_id desde Registry."""
        self.error = None
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        model_name, alias = self._parse_uri(self.model_uri)
        version = client.get_model_version_by_alias(model_name, alias)
        model = mlflow.pyfunc.load_model(self.model_uri)
        self.loaded = LoadedModel(
            model=model,
            model_name=model_name,
            model_version=str(version.version),
            model_alias=alias,
            run_id=version.run_id,
            loaded_at=datetime.now(timezone.utc),
        )
        return self.loaded

    def try_load(self) -> None:
        try:
            self.load()
        except Exception as exc:  # readiness debe responder 503, no tumbar el proceso
            self.loaded = None
            self.error = f"{type(exc).__name__}: {exc}"

    def metadata(self) -> dict[str, str]:
        if self.loaded is None:
            raise RuntimeError(self.error or "modelo no cargado")
        return {
            "model_name": self.loaded.model_name,
            "model_version": self.loaded.model_version,
            "model_alias": self.loaded.model_alias,
            "run_id": self.loaded.run_id,
            "loaded_at": self.loaded.loaded_at.isoformat(),
            "served_by": os.getenv("POD_NAME", socket.gethostname()),
        }

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        prefix = "models:/"
        if not uri.startswith(prefix) or "@" not in uri:
            raise ValueError("MODEL_URI debe tener formato models:/<nombre>@<alias>")
        name, alias = uri[len(prefix) :].split("@", 1)
        if not name or not alias:
            raise ValueError("MODEL_URI requiere nombre y alias")
        return name, alias

